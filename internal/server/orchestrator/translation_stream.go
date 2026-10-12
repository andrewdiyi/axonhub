package orchestrator

import (
	"context"
	"strings"
	"unicode/utf8"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

// translationMaxSegmentChars bounds how much assistant prose is buffered before
// it is translated and flushed, so a single very long paragraph (which never
// reaches a paragraph boundary) still streams instead of stalling the whole
// turn. Only applied outside a code fence; see shouldFlushSegment.
const translationMaxSegmentChars = 2000

// toolCallKey identifies one streamed tool call so its argument fragments can be
// grouped across chunks. A tool call is keyed by its id, falling back to the id
// learned from its identity chunk (some providers emit only the index on
// argument deltas) and finally to its position, since a response stream may
// start each message with its own tool-call list.
type toolCallKey struct {
	id string

	choice int
	index  int
}

type toolSlotKey struct {
	choice int
	index  int
}

// toolRun accumulates the argument fragments of one streamed tool call so its
// assembled JSON can be translated before the fragments are released downstream.
type toolRun struct {
	chunks []*llm.Response
	calls  []*llm.ToolCall
	name   string
}

// translatedStream lazily translates a provider's unified LLM stream as the
// client consumes it, instead of draining the whole turn first.
//
// Buffering the entire turn before emitting anything (see the previous
// implementation) meant the client received no bytes -- not even the SSE
// response headers -- until generation and translation had both finished. On a
// long turn (an agent "implement the plan" step runs for minutes) that reliably
// tripped the client's or an intermediary's idle timeout and the turn was
// cancelled. Here, assistant prose is translated one segment at a time and
// flushed as soon as a segment boundary is reached, so the client sees output
// shortly after the provider's first token while each segment still carries
// enough context to translate well.
//
// Everything that is not assistant prose -- reasoning, tool calls (except the
// user-facing interactive ones), encrypted content, and every chunk's
// transformer metadata -- is forwarded untouched and in order. The chunks are
// never rebuilt: only the text slots of a segment are rewritten in place, so a
// Responses custom tool call's script Input and a reasoning item's
// transformer metadata are never at risk.
type translatedStream struct {
	ctx      context.Context
	settings *biz.TranslationSettings
	state    *PersistenceState
	upstream streams.Stream[*llm.Response]

	// translateText and translateTool are the two translation operations, kept as
	// fields so tests can drive the stream without a live model call.
	translateText func(text string) (string, error)
	translateTool func(msg *llm.Message) bool

	out     []*llm.Response
	outHead int
	emitted []*llm.Response
	changed bool
	eof     bool

	// Current assistant-text segment awaiting translation.
	segStarted bool
	segChoice  int
	segSlots   []*string
	segChunks  []*llm.Response
	segText    strings.Builder

	// Buffered interactive/ordinary tool-call runs, keyed so fragments group.
	toolBuf   map[toolCallKey]*toolRun
	toolOrder []toolCallKey
	idByIndex map[toolSlotKey]string
}

func newTranslatedStream(
	ctx context.Context,
	caller *translationCaller,
	settings *biz.TranslationSettings,
	state *PersistenceState,
	upstream streams.Stream[*llm.Response],
) *translatedStream {
	return &translatedStream{
		ctx:      ctx,
		settings: settings,
		state:    state,
		upstream: upstream,
		translateText: func(text string) (string, error) {
			return caller.translate(ctx, settings.ChannelID, settings.Model, text, settings.OutgoingPromptTemplate, settings.HumanLanguage)
		},
		translateTool: func(msg *llm.Message) bool {
			return translateInteractiveToolCalls(ctx, caller, msg, settings.ChannelID, settings.Model, settings.OutgoingPromptTemplate, settings.HumanLanguage)
		},
		toolBuf:   map[toolCallKey]*toolRun{},
		idByIndex: map[toolSlotKey]string{},
	}
}

// Next advances to the next chunk the client should receive, buffering from the
// upstream only until at least one chunk is ready. A chunk that carries user
// prose is not released until its whole segment has been translated, so this may
// read several upstream chunks before returning; every other chunk returns after
// a single read, preserving streaming.
func (s *translatedStream) Next() bool {
	if s.outHead < len(s.out) {
		return true
	}

	if len(s.out) > 0 {
		s.out = s.out[:0]
		s.outHead = 0
	}

	for len(s.out) == 0 {
		if s.eof {
			return false
		}

		if s.upstream == nil {
			s.eof = true

			return false
		}

		if !s.upstream.Next() {
			s.eof = true

			// End of the source: release whatever is still buffered, then persist
			// the client-facing (translated) turn for audit.
			s.flushSegment()
			s.flushAllTools()
			s.persist()

			continue
		}

		s.handle(s.upstream.Current())
	}

	return true
}

// Current returns the current chunk and advances the read cursor, mirroring the
// standard Next/Current stream contract used elsewhere in the codebase.
func (s *translatedStream) Current() *llm.Response {
	if s.outHead >= len(s.out) {
		return nil
	}

	chunk := s.out[s.outHead]
	s.outHead++

	return chunk
}

func (s *translatedStream) Err() error {
	if s.upstream == nil {
		return nil
	}

	return s.upstream.Err()
}

func (s *translatedStream) Close() error {
	if s.upstream == nil {
		return nil
	}

	return s.upstream.Close()
}

// handle routes one upstream chunk: it either extends the current text segment,
// opens/buffers a tool-call run, or is forwarded as-is. Forwarding order is
// preserved by flushing the current text segment before any non-text chunk is
// emitted.
func (s *translatedStream) handle(chunk *llm.Response) {
	if chunk == nil {
		return
	}

	if len(chunk.Choices) == 0 {
		// Terminal sentinel ([DONE]) and other choice-less events pass through.
		s.emit(chunk)

		return
	}

	choice := &chunk.Choices[0]
	delta := choice.Delta

	if delta != nil && (len(delta.ToolCalls) > 0 || deltaHasNonTextOutput(delta)) {
		// A tool call or reasoning/non-text output ends the current text run. Its
		// own text (if any) is emitted untranslated, which in practice never
		// happens: provider streams keep text and tool/reasoning chunks disjoint.
		s.flushSegment()

		if len(delta.ToolCalls) > 0 {
			s.bufferToolChunk(chunk, choice.Index, delta)
		} else {
			s.emit(chunk)
		}

		return
	}

	if delta != nil && translationRoleMatches(cohesiveRole(delta.Role)) {
		if slot, tail, ok := textSlotsOf(delta); ok {
			if s.segStarted && s.segChoice != choice.Index {
				s.flushSegment()
			}

			s.segStarted = true
			s.segChoice = choice.Index

			if slot != nil {
				s.segSlots = append(s.segSlots, slot)
				s.segText.WriteString(*slot)
			}

			for _, t := range tail {
				s.segSlots = append(s.segSlots, t)
				s.segText.WriteString(*t)
			}

			s.segChunks = append(s.segChunks, chunk)

			if s.shouldFlushSegment() {
				s.flushSegment()
			}

			return
		}
	}

	s.emit(chunk)
}

// shouldFlushSegment reports whether the buffered text is at a release point:
// a true paragraph boundary outside a code fence, or the size cap (also outside
// a fence, so a long code block is not split and sent to the model mid-block).
func (s *translatedStream) shouldFlushSegment() bool {
	if s.segText.Len() == 0 {
		return false
	}

	text := s.segText.String()
	if codeFenceOpen(text) {
		return false
	}

	return isParagraphBoundary(text) || len(text) >= translationMaxSegmentChars
}

// isParagraphBoundary reports whether the buffered text ends at a true paragraph
// boundary: its last content line closes a sentence (or closes a code fence) and
// is followed by a real blank line. Only the buffer's end is examined -- cutting
// there never strands a group, since the prefix stays whole, and a line held
// mid-buffer is still present when the boundary is finally reached.
//
// A blank line whose preceding content does not end a sentence (a markdown list
// item, an address line, a heading with no terminal punctuation, ...) is a soft
// break that renders as part of the same flowing block, so the segment is held
// until a real paragraph end or the size cap. This keeps a heading with the
// paragraph it introduces and a list with its items.
func isParagraphBoundary(text string) bool {
	lines := strings.Split(text, "\n")

	last := len(lines) - 1
	for last >= 0 && strings.TrimSpace(lines[last]) == "" {
		last--
	}

	// A real blank line means two newlines after the last content line; a single
	// trailing newline merely ends that line.
	if last < 0 || len(lines)-1-last < 2 {
		return false
	}

	// A closing code fence is a boundary on its own, with no terminal punctuation.
	if strings.HasPrefix(strings.TrimSpace(lines[last]), "```") {
		return true
	}

	return endsSentence(lines[last])
}

// sentenceTrailingMarkup are trailing decorations stripped before checking for
// terminal punctuation (emphasis, code spans, closing brackets/quotes).
var sentenceTrailingMarkup = []string{"**", "__", "*", "_", "`", "”", "’", "\"", ")", "]", "}"}

// endsSentence reports whether the line ends with terminal punctuation that
// closes a sentence, tolerating trailing markup.
func endsSentence(line string) bool {
	line = strings.TrimRight(line, " \t")

	for {
		trimmed := false

		for _, suffix := range sentenceTrailingMarkup {
			if strings.HasSuffix(line, suffix) {
				line = strings.TrimRight(strings.TrimSuffix(line, suffix), " \t")
				trimmed = true

				break
			}
		}

		if !trimmed {
			break
		}
	}

	if line == "" {
		return false
	}

	r, _ := utf8.DecodeLastRuneInString(line)

	switch r {
	case '.', '!', '?', '。', '！', '？', '…', '‽':
		return true
	default:
		return false
	}
}

// flushSegment translates the buffered assistant text (if any) and emits the
// chunks that carried it, rewriting the first text slot and emptying the rest so
// downstream transformers, which concatenate deltas in order, still deliver the
// whole translation.
func (s *translatedStream) flushSegment() {
	if !s.segStarted || len(s.segChunks) == 0 {
		s.resetSegment()

		return
	}

	text := s.segText.String()
	if shouldTranslate(text, s.settings.HumanLanguage) {
		translated, err := s.translateText(text)
		switch {
		case err != nil:
			log.Warn(s.ctx, "failed to translate outgoing streamed response, passing through original text", log.Cause(err))
		case translated != text:
			*s.segSlots[0] = translated

			for _, slot := range s.segSlots[1:] {
				*slot = ""
			}

			s.changed = true
		}
	}

	for _, chunk := range s.segChunks {
		s.emit(chunk)
	}

	s.resetSegment()
}

func (s *translatedStream) resetSegment() {
	s.segStarted = false
	s.segSlots = nil
	s.segChunks = nil
	s.segText.Reset()
}

// bufferToolChunk groups one chunk's tool-call fragment under its call key,
// flushing any buffered run the chunk does not continue so runs stay ordered.
// Only the first tool call of a chunk is tracked; provider streams carry one
// tool call's fragment per chunk.
func (s *translatedStream) bufferToolChunk(chunk *llm.Response, choiceIndex int, delta *llm.Message) {
	key := s.keyFor(choiceIndex, &delta.ToolCalls[0])

	for _, pending := range s.toolOrder {
		if pending == key {
			continue
		}

		if _, ok := s.toolBuf[pending]; ok {
			s.flushTool(pending)
		}
	}

	run := s.toolBuf[key]
	if run == nil {
		run = &toolRun{}
		s.toolBuf[key] = run
		s.toolOrder = append(s.toolOrder, key)
	}

	if delta.ToolCalls[0].Function.Name != "" {
		run.name = delta.ToolCalls[0].Function.Name
	}

	run.chunks = append(run.chunks, chunk)

	for j := range delta.ToolCalls {
		run.calls = append(run.calls, &delta.ToolCalls[j])
	}
}

func (s *translatedStream) flushAllTools() {
	for _, key := range s.toolOrder {
		if _, ok := s.toolBuf[key]; ok {
			s.flushTool(key)
		}
	}

	s.toolOrder = nil
}

func (s *translatedStream) flushTool(key toolCallKey) {
	run := s.toolBuf[key]
	if run == nil {
		return
	}

	delete(s.toolBuf, key)

	if _, ok := interactiveToolNames[strings.ToLower(run.name)]; ok {
		if s.translateToolRun(run) {
			s.changed = true
		}
	}

	for _, chunk := range run.chunks {
		s.emit(chunk)
	}
}

// translateToolRun assembles a streamed interactive tool call's argument
// fragments, translates the user-facing prose inside them, and writes the result
// back into the first fragment (emptying the rest), returning whether anything
// changed.
func (s *translatedStream) translateToolRun(run *toolRun) bool {
	if len(run.calls) == 0 {
		return false
	}

	var builder strings.Builder

	for _, call := range run.calls {
		builder.WriteString(call.Function.Arguments)
	}

	args := builder.String()
	if strings.TrimSpace(args) == "" {
		return false
	}

	msg := &llm.Message{ToolCalls: []llm.ToolCall{{
		Type:     run.calls[0].Type,
		Index:    run.calls[0].Index,
		Function: llm.FunctionCall{Name: run.name, Arguments: args},
	}}}

	if !s.translateTool(msg) {
		return false
	}

	run.calls[0].Function.Arguments = msg.ToolCalls[0].Function.Arguments
	for _, call := range run.calls[1:] {
		call.Function.Arguments = ""
	}

	return true
}

func (s *translatedStream) keyFor(choiceIndex int, tc *llm.ToolCall) toolCallKey {
	slot := toolSlotKey{choice: choiceIndex, index: tc.Index}

	if tc.ID != "" {
		s.idByIndex[slot] = tc.ID

		return toolCallKey{id: tc.ID}
	}

	if id, ok := s.idByIndex[slot]; ok && id != "" {
		return toolCallKey{id: id}
	}

	return toolCallKey{choice: choiceIndex, index: tc.Index}
}

func (s *translatedStream) emit(chunk *llm.Response) {
	s.out = append(s.out, chunk)
	s.emitted = append(s.emitted, chunk)
}

func (s *translatedStream) persist() {
	if s.changed {
		persistTranslatedStream(s.ctx, s.state, s.emitted)
	}
}

// textSlotsOf returns the writable text slots a delta carries -- the scalar
// Content plus any "text"-typed MultipleContent parts -- in the order downstream
// transformers concatenate them.
func textSlotsOf(delta *llm.Message) (*string, []*string, bool) {
	var (
		slot *string
		tail []*string
	)

	if delta.Content.Content != nil {
		slot = delta.Content.Content
	}

	for i := range delta.Content.MultipleContent {
		part := &delta.Content.MultipleContent[i]
		if strings.EqualFold(part.Type, "text") && part.Text != nil {
			tail = append(tail, part.Text)
		}
	}

	return slot, tail, slot != nil || len(tail) > 0
}

// deltaHasNonTextOutput reports whether a delta carries output other than
// assistant text (and other than tool calls, which the caller checks
// separately): reasoning, inline tool results, or a non-text content part.
func deltaHasNonTextOutput(delta *llm.Message) bool {
	if delta.ReasoningContent != nil || delta.Reasoning != nil ||
		delta.ReasoningSignature != nil || len(delta.InlineToolResults) > 0 {
		return true
	}

	for i := range delta.Content.MultipleContent {
		if !strings.EqualFold(delta.Content.MultipleContent[i].Type, "text") {
			return true
		}
	}

	return false
}

// codeFenceOpen reports whether a ``` fence is left unclosed in text, so a
// blank line inside a code block is not mistaken for a paragraph boundary.
func codeFenceOpen(text string) bool {
	open := false

	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			open = !open
		}
	}

	return open
}
