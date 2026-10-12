package orchestrator

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

// collect drains a translatedStream into the chunks a client would receive.
func collect(s *translatedStream) []*llm.Response {
	var got []*llm.Response

	for s.Next() {
		got = append(got, s.Current())
	}

	return got
}

// textChunk builds a streamed assistant-text delta.
func textChunk(content string) *llm.Response {
	return &llm.Response{Choices: []llm.Choice{{
		Index: 0,
		Delta: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: strptr(content)}},
	}}}
}

// toolArgsChunk builds a streamed tool-call argument fragment.
func toolArgsChunk(name string, index int, args string) *llm.Response {
	return &llm.Response{Choices: []llm.Choice{{
		Index: 0,
		Delta: &llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{
			Index:    index,
			Function: llm.FunctionCall{Name: name, Arguments: args},
		}}},
	}}}
}

func strptr(s string) *string { return &s }
func streamOf(chunks ...*llm.Response) streams.Stream[*llm.Response] {
	return streams.SliceStream(chunks)
}

func testSettings() *biz.TranslationSettings {
	return &biz.TranslationSettings{Enabled: true, HumanLanguage: "简体中文", AgentLanguage: "English", ChannelID: 30, Model: "deepseek-flash"}
}

// prefixed wraps upstream translation to something recognizable so the test can
// assert the translated text was actually written into the stream.
func prefixed() func(string) (string, error) {
	return func(text string) (string, error) { return "zh:" + text, nil }
}

// TestTranslatedStreamFlushesBeforeEOF proves the wrapper does not buffer the
// whole turn: a segment boundary releases a translated chunk before the upstream
// has been drained, which is what lets the client receive bytes early instead of
// waiting out a long turn (and being timed out).
func TestTranslatedStreamFlushesBeforeEOF(t *testing.T) {
	upstream := streamOf(
		textChunk("第一段。\n\n"),
		textChunk("第二段。"),
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := &translatedStream{
		ctx:           ctx,
		settings:      testSettings(),
		upstream:      upstream,
		translateText: prefixed(),
		toolBuf:       map[toolCallKey]*toolRun{},
		idByIndex:     map[toolSlotKey]string{},
	}

	if !s.Next() {
		t.Fatal("expected a first chunk before the upstream was drained")
	}

	if got := *s.Current().Choices[0].Delta.Content.Content; got != "zh:第一段。\n\n" {
		t.Fatalf("first chunk = %q, want the translated first paragraph", got)
	}

	rest := collect(s)
	if len(rest) != 1 {
		t.Fatalf("remaining chunks = %d, want 1", len(rest))
	}

	if got := *rest[0].Choices[0].Delta.Content.Content; got != "zh:第二段。" {
		t.Fatalf("tail = %q, want translated final segment", got)
	}
}

// TestTranslatedStreamSkipsBlankLineInsideFence verifies a blank line inside an
// open code fence does not split the segment, so a code block is never sent to
// the translator in halves.
func TestTranslatedStreamSkipsBlankLineInsideFence(t *testing.T) {
	fenced := "```js\nconst a = 1;\n\nconst b = 2;\n```\n\n"
	upstream := streamOf(textChunk(fenced))

	s := &translatedStream{
		ctx:           context.Background(),
		settings:      testSettings(),
		upstream:      upstream,
		translateText: prefixed(),
		toolBuf:       map[toolCallKey]*toolRun{},
		idByIndex:     map[toolSlotKey]string{},
	}

	// The whole fenced block plus its trailing blank line flush as one segment;
	// the blank line inside the fence must not have split it into two calls.
	chunks := collect(s)
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1 (fence kept whole)", len(chunks))
	}

	if got := *chunks[0].Choices[0].Delta.Content.Content; got != "zh:"+fenced {
		t.Fatalf("got %q", got)
	}
}

// TestTranslatedStreamTranslatesStreamedInteractiveTool verifies a streamed
// interactive tool call's argument fragments are grouped, translated as one
// JSON payload, and written back into the first fragment.
func TestTranslatedStreamTranslatesStreamedInteractiveTool(t *testing.T) {
	upstream := streamOf(
		toolArgsChunk("request_user_input", 0, `{"questions":[{"header":"3D`),
		toolArgsChunk("request_user_input", 0, ` model","question":"Which?"}]}`),
	)

	s := &translatedStream{
		ctx:      context.Background(),
		settings: testSettings(),
		upstream: upstream,
		translateText: func(text string) (string, error) {
			return text, nil
		},
		translateTool: func(msg *llm.Message) bool {
			// Mirror the real behavior: rewrite the header field.
			args := msg.ToolCalls[0].Function.Arguments
			msg.ToolCalls[0].Function.Arguments = strings.ReplaceAll(args, "3D model", "3D模型")

			return true
		},
		toolBuf:   map[toolCallKey]*toolRun{},
		idByIndex: map[toolSlotKey]string{},
		toolNames: map[toolSlotKey]string{},
	}

	chunks := collect(s)
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2 (both fragments forwarded)", len(chunks))
	}

	first := chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments
	second := chunks[1].Choices[0].Delta.ToolCalls[0].Function.Arguments

	combined := first + second

	var parsed struct {
		Questions []struct {
			Header string `json:"header"`
		} `json:"questions"`
	}

	if err := json.Unmarshal([]byte(combined), &parsed); err != nil {
		t.Fatalf("combined arguments are not valid JSON %q: %v", combined, err)
	}

	if parsed.Questions[0].Header != "3D模型" {
		t.Fatalf("header = %q, want translated", parsed.Questions[0].Header)
	}
}

// TestTranslatedStreamLeavesNonInteractiveToolUntouched verifies ordinary tool
// calls (whose arguments are machine data) are forwarded verbatim.
func TestTranslatedStreamLeavesNonInteractiveToolUntouched(t *testing.T) {
	upstream := streamOf(toolArgsChunk("exec", 0, `{"cmd":"ls -la"}`))

	s := &translatedStream{
		ctx:      context.Background(),
		settings: testSettings(),
		upstream: upstream,
		translateText: func(text string) (string, error) {
			return text, nil
		},
		translateTool: func(msg *llm.Message) bool {
			t.Fatal("non-interactive tool must not be translated")

			return false
		},
		toolBuf:   map[toolCallKey]*toolRun{},
		idByIndex: map[toolSlotKey]string{},
		toolNames: map[toolSlotKey]string{},
	}

	chunks := collect(s)
	if got := chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments; got != `{"cmd":"ls -la"}` {
		t.Fatalf("exec args = %q, want untouched", got)
	}
}

// TestTranslatedStreamForwardsCustomToolIdentityImmediately guards the regression
// where a Responses custom tool call's opening chunk (which alone carries the
// call id/name) was buffered while only its later argument deltas were released,
// stripping the provider request of the call identity. The identity chunk must
// reach the client before any argument delta.
func TestTranslatedStreamForwardsCustomToolIdentityImmediately(t *testing.T) {
	identity := func() *llm.Response {
		return &llm.Response{Choices: []llm.Choice{{Index: 0, Delta: &llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{{
				ID:   "call_abc",
				Type: "responses_custom_tool",
				ResponseCustomToolCall: &llm.ResponseCustomToolCall{
					CallID: "call_abc",
					Name:   "exec",
				},
			}},
		}}}}
	}

	argDelta := func(input string) *llm.Response {
		return &llm.Response{Choices: []llm.Choice{{Index: 0, Delta: &llm.Message{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{{
				Index: 0,
				Type:  "responses_custom_tool",
				ResponseCustomToolCall: &llm.ResponseCustomToolCall{
					CallID: "call_abc",
					Name:   "exec",
					Input:  input,
				},
			}},
		}}}}
	}

	upstream := streamOf(
		textChunk("我先看一下。"),
		identity(),
		argDelta("text(await"),
		argDelta(" tools.exec());"),
	)

	s := &translatedStream{
		ctx:           context.Background(),
		settings:      testSettings(),
		upstream:      upstream,
		translateText: func(text string) (string, error) { return text, nil },
		translateTool: func(msg *llm.Message) bool { return false },
		toolBuf:       map[toolCallKey]*toolRun{},
		idByIndex:     map[toolSlotKey]string{},
		toolNames:     map[toolSlotKey]string{},
	}

	chunks := collect(s)

	// The identity chunk (with the call id) must appear, and before any argument
	// fragment; otherwise the client never learns what tool to run.
	identityIdx, firstArgIdx := -1, -1

	for i, c := range chunks {
		tc := c.Choices[0].Delta.ToolCalls
		if len(tc) == 0 {
			continue
		}

		if tc[0].ResponseCustomToolCall != nil && tc[0].ResponseCustomToolCall.Input == "" {
			if identityIdx == -1 {
				identityIdx = i
			}
		} else if identityIdx != -1 && firstArgIdx == -1 {
			firstArgIdx = i
		}
	}

	if identityIdx == -1 {
		t.Fatal("identity chunk (call id) was not forwarded")
	}

	if firstArgIdx != -1 && firstArgIdx < identityIdx {
		t.Fatalf("argument delta forwarded before identity chunk (arg=%d id=%d)", firstArgIdx, identityIdx)
	}
}

// TestTranslatedStreamHoldsBlankLineAfterListItem verifies a blank line after a
// markdown list item (no terminal punctuation) is not treated as a paragraph
// boundary, so the list's items are translated together rather than one at a
// time without each other's context.
func TestTranslatedStreamHoldsBlankLineAfterListItem(t *testing.T) {
	list := "- 第一项\n\n- 第二项。\n\n"

	upstream := streamOf(textChunk(list))

	s := &translatedStream{
		ctx:           context.Background(),
		settings:      testSettings(),
		upstream:      upstream,
		translateText: prefixed(),
		toolBuf:       map[toolCallKey]*toolRun{},
		idByIndex:     map[toolSlotKey]string{},
	}

	// Only one translation, over the whole list, is issued at EOF.
	chunks := collect(s)
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1 (list kept together)", len(chunks))
	}

	if got := *chunks[0].Choices[0].Delta.Content.Content; got != "zh:"+list {
		t.Fatalf("got %q, want the whole list translated once", got)
	}
}

// TestIsParagraphBoundary covers the paragraph-boundary classifier directly.
func TestIsParagraphBoundary(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
	}{
		{"paragraph end (chinese)", "第一段。\n\n", true},
		{"paragraph end (english)", "First.\n\n", true},
		{"no blank line", "one long line", false},
		{"one trailing newline only", "First.\n", false},
		{"list item break", "- item\n\n", false},
		{"heading break", "## 标题\n\n", false},
		{"heading then paragraph", "## Title\n\nBody.\n\n", true},
		{"fenced block kept separate", "```js\ncode\n```\n\n", true},
		{"markup after period", "- **Done.**\n\n", true},
		{"question then break", "Which?\n\n", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isParagraphBoundary(tc.text); got != tc.want {
				t.Fatalf("isParagraphBoundary(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

// TestTranslatedStreamClearsTrailingSlots verifies that when a segment's text
// spans several slots, the translation lands in the first and the rest are
// emptied, so downstream concatenation still yields the whole translation once.
func TestTranslatedStreamClearsTrailingSlots(t *testing.T) {
	first := "hello "
	second := "world"

	upstream := streamOf(
		&llm.Response{Choices: []llm.Choice{{Index: 0, Delta: &llm.Message{
			Role: "assistant",
			Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{
				{Type: "text", Text: &first},
				{Type: "text", Text: &second},
			}},
		}}}},
	)

	s := &translatedStream{
		ctx:           context.Background(),
		settings:      testSettings(),
		upstream:      upstream,
		translateText: func(text string) (string, error) { return "ZH", nil },
		toolBuf:       map[toolCallKey]*toolRun{},
		idByIndex:     map[toolSlotKey]string{},
	}

	chunks := collect(s)
	parts := chunks[0].Choices[0].Delta.Content.MultipleContent

	if got := *parts[0].Text; got != "ZH" {
		t.Fatalf("first slot = %q, want ZH", got)
	}

	if got := *parts[1].Text; got != "" {
		t.Fatalf("second slot = %q, want emptied", got)
	}
}
