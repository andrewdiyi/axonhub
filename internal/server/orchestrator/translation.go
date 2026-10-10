package orchestrator

import (
	"context"
	"encoding/json"
	"sort"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
)

// translateMessages registers the automatic request/response translation
// middleware. It is always appended to the pipeline (matching protectPrompts'
// pattern); the middleware itself no-ops when translation is disabled, when
// no translation caller is available, or when it is handling the translation
// caller's own internal request (see isInternalTranslationCall).
func translateMessages(inbound *PersistentInboundTransformer, outbound *PersistentOutboundTransformer) pipeline.Middleware {
	return &translationMiddleware{inbound: inbound, outbound: outbound}
}

type translationMiddleware struct {
	pipeline.DummyMiddleware

	inbound  *PersistentInboundTransformer
	outbound *PersistentOutboundTransformer
}

func (m *translationMiddleware) Name() string {
	return "translate-messages"
}

// OnInboundLlmRequest normalizes the conversation history toward the agent
// language before the request reaches the upstream provider. The client resends
// the full history every turn in the human language, so every scoped message is
// translated toward the agent language; carried-over messages hit the result
// cache instead of being retranslated. See normalizeHistoryTowardAgent.
func (m *translationMiddleware) OnInboundLlmRequest(ctx context.Context, request *llm.Request) (*llm.Request, error) {
	if isInternalTranslationCall(ctx) {
		return request, nil
	}

	caller := getTranslationCaller()
	if caller == nil {
		return request, nil
	}

	settings := m.inbound.state.SystemService.TranslationSettingsOrDefault(ctx)
	if !settings.Enabled || settings.ChannelID == 0 || settings.Model == "" {
		return request, nil
	}

	// Structured-output requests are agent-harness internal jobs (e.g. generating
	// a task title) that embed the user's text after a marker inside a larger,
	// English instruction block. Translating the whole message would garble the
	// instructions (observed: the model echoed the instructions back instead of
	// translating), so only the embedded user text is translated.
	if hasStructuredOutputConstraint(request) {
		if translateStructuredOutputTail(ctx, caller, request, settings) {
			m.inbound.state.TranslationApplied = true
		}

		return request, nil
	}

	index := lastTranslatableMessageIndex(request.Messages)
	if index < 0 {
		return request, nil
	}

	// Normalize the whole conversation history so the model always sees one
	// consistent language. The client resends the full history every turn, and
	// that history is written in the human language (the user's originals and the
	// translated replies it received). Replaying it verbatim would put the human
	// language next to the agent-language text we send, so the model sees a
	// "the user wrote X but I answered Y" mismatch and contradicts itself. Every
	// resent message is therefore translated toward the agent language; the
	// current message is already the tail of that history.
	if normalizeHistoryTowardAgent(ctx, caller, request.Messages[:index+1], settings) {
		// Only suppress body pass-through when the text actually changed. A no-op
		// (translation returned the same text, e.g. already agent-language, or the
		// message had no text) must not disable pass-through.
		m.inbound.state.TranslationApplied = true
	}

	return request, nil
}

// normalizeHistoryTowardAgent translates every scoped message in the conversation
// history toward the agent language. Each message is translated through the
// caller's result cache keyed on its own text, so a message carried over from a
// previous turn (the client resends the whole history every turn) is served from
// cache instead of being retranslated, and always maps to the same translation.
// Returns true when any message's text actually changed. Best-effort: a message
// that fails to translate keeps its original text.
func normalizeHistoryTowardAgent(ctx context.Context, caller *translationCaller, messages []llm.Message, settings *biz.TranslationSettings) bool {
	before := make([]string, len(messages))
	for i := range messages {
		before[i] = messageTextFingerprint(&messages[i])
	}

	for i := range messages {
		msg := &messages[i]
		if !translationRoleMatches(msg.Role) {
			continue
		}

		if err := translateMessageContent(ctx, caller, msg, settings.ChannelID, settings.Model, settings.IncomingPromptTemplate, settings.AgentLanguage); err != nil {
			log.Warn(ctx, "failed to translate history message, keeping original", log.Cause(err))
		}
	}

	changed := false

	for i := range messages {
		if messageTextFingerprint(&messages[i]) != before[i] {
			changed = true

			break
		}
	}

	return changed
}

// messageTextFingerprint returns the message's translatable text segments joined
// into one string, used to detect whether translation actually changed anything.
func messageTextFingerprint(msg *llm.Message) string {
	var builder strings.Builder

	if msg.Content.Content != nil {
		builder.WriteString(*msg.Content.Content)
		builder.WriteByte(0)
	}

	for i := range msg.Content.MultipleContent {
		part := &msg.Content.MultipleContent[i]
		if !strings.EqualFold(part.Type, "text") || part.Text == nil {
			continue
		}

		builder.WriteString(*part.Text)
		builder.WriteByte(0)
	}

	return builder.String()
}

// OnOutboundLlmResponse translates the non-streaming assistant reply toward
// HumanLanguage before it is returned to the client, and records the
// translated text alongside the already-persisted original response body.
func (m *translationMiddleware) OnOutboundLlmResponse(ctx context.Context, response *llm.Response) (*llm.Response, error) {
	if isInternalTranslationCall(ctx) || response == nil || response.Error != nil {
		return response, nil
	}

	caller := getTranslationCaller()
	if caller == nil {
		return response, nil
	}

	state := m.outbound.state
	settings := state.SystemService.TranslationSettingsOrDefault(ctx)
	if !settings.Enabled || settings.ChannelID == 0 || settings.Model == "" {
		return response, nil
	}

	// Structured-output replies are machine data (e.g. {"title": ...}) shaped by a
	// JSON schema, not prose. Only the request's embedded user text is translated
	// (see translateStructuredOutputTail); the reply is left as the harness's
	// schema-compliant original.
	if state.LlmRequest != nil && hasStructuredOutputConstraint(state.LlmRequest) {
		return response, nil
	}

	translated := false
	for i := range response.Choices {
		msg := response.Choices[i].Message
		if msg == nil {
			continue
		}

		role := msg.Role
		if role == "" {
			role = "assistant"
		}

		if !translationRoleMatches(role) {
			continue
		}

		translatedThisMessage := false

		if err := translateMessageContent(ctx, caller, msg, settings.ChannelID, settings.Model, settings.OutgoingPromptTemplate, settings.HumanLanguage); err == nil {
			translatedThisMessage = true
		} else {
			log.Warn(ctx, "failed to translate outgoing response, passing through original text", log.Cause(err))
		}

		// Interactive tools (e.g. Codex plan mode's request_user_input) carry the
		// questions and options the human reads, so their prose arguments are
		// translated too. Ordinary tool calls are left untouched.
		if translateInteractiveToolCalls(ctx, caller, msg, settings.ChannelID, settings.Model, settings.OutgoingPromptTemplate, settings.HumanLanguage) {
			translatedThisMessage = true
		}

		if translatedThisMessage {
			translated = true
		}
	}

	if translated {
		persistTranslatedResponse(ctx, state, response)
	}

	return response, nil
}

// OnOutboundLlmStream buffers the full streamed reply, translates the
// user-facing prose in it once (there is no hook that exposes the full text of a
// streamed turn before delivery, so this is the simplest correct behavior), and
// then re-emits the stream with those prose deltas rewritten in place. Every
// other byte -- reasoning, tool calls, encrypted content, and each chunk's
// transformer metadata -- is passed through untouched. This trades incremental
// streaming UX for correctness: the client receives nothing until generation and
// translation both complete.
func (m *translationMiddleware) OnOutboundLlmStream(ctx context.Context, stream streams.Stream[*llm.Response]) (streams.Stream[*llm.Response], error) {
	if isInternalTranslationCall(ctx) {
		return stream, nil
	}

	caller := getTranslationCaller()
	if caller == nil {
		return stream, nil
	}

	state := m.outbound.state
	settings := state.SystemService.TranslationSettingsOrDefault(ctx)
	if !settings.Enabled || settings.ChannelID == 0 || settings.Model == "" {
		return stream, nil
	}

	// Translating a structured-output reply would break its JSON schema, so only
	// the request's embedded user text is translated (see
	// translateStructuredOutputTail); the reply is left untouched.
	if state.LlmRequest != nil && hasStructuredOutputConstraint(state.LlmRequest) {
		return stream, nil
	}

	// streams.All drains but does not close. Every other wrapper (Map, Filter,
	// ...) delegates Close() to the stream it wraps, so the eventual consumer's
	// Close() cascades down to OutboundPersistentStream.Close(), where
	// response-chunk persistence and execution-status finalization happen.
	// Returning an unrelated streams.SliceStream below would otherwise sever
	// that chain, leaving the execution stuck in "processing" forever.
	defer stream.Close()

	chunks, err := streams.All(stream)
	if err != nil {
		return streams.SliceStream(chunks), err
	}

	if translateOutgoingStreamChunks(ctx, caller, chunks, settings) {
		persistTranslatedStream(ctx, state, chunks)
	}

	return streams.SliceStream(chunks), nil
}

// translateOutgoingStreamChunks rewrites, in place, the user-facing prose a
// buffered streamed turn carries -- each choice's assistant text and the prose
// arguments of interactive tools -- and reports whether anything changed.
// Everything else (reasoning, ordinary tool calls including a Responses custom
// tool call's script Input, encrypted content, and every chunk's transformer
// metadata) is left exactly as the provider emitted it.
//
// Rebuilding the turn from a merged message instead silently drops data the
// client depends on: merging only Function.Arguments emptied a Responses custom
// tool call's Input (the script the agent must run, so the next turn failed with
// "expects raw JavaScript source text"), and collapsing the chunks kept only the
// last chunk's transformer metadata, detaching a reasoning item's encrypted
// content from its provider id (the next turn then replayed an id the upstream
// could not verify). Rewriting just the text in place reconstructs nothing.
func translateOutgoingStreamChunks(ctx context.Context, caller *translationCaller, chunks []*llm.Response, settings *biz.TranslationSettings) bool {
	// A streamed choice's text arrives split across many chunks. Translate each
	// contiguous run of text as one string (translating fragments separately
	// loses the sentence context the model needs), then write the result into the
	// run's first text slot and empty the rest: downstream transformers
	// concatenate the deltas in order, so the client still reads the whole
	// translation. A run ends when a chunk carrying a tool call or reasoning sits
	// between two text chunks, so a turn that interleaves prose with tool calls
	// keeps its original ordering instead of being collapsed to the front. Runs
	// are tracked per choice index so a multi-choice stream never mixes them.
	type textRun struct {
		slots  []*string
		broken bool
	}

	runs := map[int]*textRun{}
	var (
		choiceOrder []int
		textRuns    [][]*string
	)

	// Interactive tool arguments stream across chunks too; group the chunks
	// carrying them so the tool's JSON can be assembled before its prose is
	// translated. A tool call is keyed by its id, falling back to the id learned
	// from its identity chunk (some providers emit only the index on argument
	// deltas) and finally to its position: a response stream may start each
	// message with its own tool-call list, so "first tool call seen" is not stable.
	type toolKey struct {
		id string

		choice int
		index  int
	}

	type slotKey struct {
		choice int
		index  int
	}

	toolSlots := map[toolKey][]*llm.ToolCall{}
	toolNames := map[toolKey]string{}
	idByIndex := map[slotKey]string{}

	keyFor := func(choiceIndex int, tc *llm.ToolCall) toolKey {
		slot := slotKey{choice: choiceIndex, index: tc.Index}

		if tc.ID != "" {
			idByIndex[slot] = tc.ID

			return toolKey{id: tc.ID}
		}

		if id, ok := idByIndex[slot]; ok && id != "" {
			return toolKey{id: id}
		}

		return toolKey{choice: choiceIndex, index: tc.Index}
	}

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}

		for i := range chunk.Choices {
			choice := &chunk.Choices[i]
			delta := choice.Delta
			if delta == nil {
				continue
			}

			run := runs[choice.Index]
			if run == nil {
				run = &textRun{}
				runs[choice.Index] = run
				choiceOrder = append(choiceOrder, choice.Index)
			}

			// Any non-text output ends the current text run.
			if len(delta.ToolCalls) > 0 || delta.ReasoningContent != nil || delta.Reasoning != nil ||
				delta.ReasoningSignature != nil || len(delta.InlineToolResults) > 0 {
				run.broken = true
			}

			var (
				slot      *string
				tailParts []*string
			)

			if translationRoleMatches(cohesiveRole(delta.Role)) {
				if delta.Content.Content != nil {
					slot = delta.Content.Content
				}

				for j := range delta.Content.MultipleContent {
					part := &delta.Content.MultipleContent[j]
					if strings.EqualFold(part.Type, "text") && part.Text != nil {
						tailParts = append(tailParts, part.Text)

						continue
					}

					// A non-text part (image, compaction summary, ...) is a separate
					// output item; text after it must not merge into the run before it.
					run.broken = true
				}
			}

			if slot != nil || len(tailParts) > 0 {
				if run.broken && len(run.slots) > 0 {
					textRuns = append(textRuns, run.slots)
					run.slots = nil
				}

				run.broken = false

				if slot != nil {
					run.slots = append(run.slots, slot)
				}

				run.slots = append(run.slots, tailParts...)
			}

			for j := range delta.ToolCalls {
				tc := &delta.ToolCalls[j]
				key := keyFor(choice.Index, tc)

				if tc.Function.Name != "" {
					toolNames[key] = tc.Function.Name
				}

				if tc.Function.Arguments != "" {
					toolSlots[key] = append(toolSlots[key], tc)
				}
			}
		}
	}

	for _, index := range choiceOrder {
		run := runs[index]
		if len(run.slots) > 0 {
			textRuns = append(textRuns, run.slots)
		}
	}

	changed := false

	for _, slots := range textRuns {
		var builder strings.Builder

		for _, slot := range slots {
			builder.WriteString(*slot)
		}

		text := builder.String()
		if !shouldTranslate(text, settings.HumanLanguage) {
			continue
		}

		translated, err := caller.translate(ctx, settings.ChannelID, settings.Model, text, settings.OutgoingPromptTemplate, settings.HumanLanguage)
		if err != nil {
			log.Warn(ctx, "failed to translate outgoing streamed response, passing through original text", log.Cause(err))

			continue
		}

		if translated == text {
			continue
		}

		*slots[0] = translated
		for _, slot := range slots[1:] {
			*slot = ""
		}

		changed = true
	}

	for key, slots := range toolSlots {
		if _, ok := interactiveToolNames[strings.ToLower(toolNames[key])]; !ok {
			continue
		}

		var builder strings.Builder

		for _, tc := range slots {
			builder.WriteString(tc.Function.Arguments)
		}

		// Reuse the non-streaming interactive-tool translation by assembling the
		// streamed fragments into one tool call, translating it, then writing the
		// result back into the first fragment's slot.
		msg := &llm.Message{ToolCalls: []llm.ToolCall{{
			Type:     slots[0].Type,
			Index:    slots[0].Index,
			Function: llm.FunctionCall{Name: toolNames[key], Arguments: builder.String()},
		}}}

		if !translateInteractiveToolCalls(ctx, caller, msg, settings.ChannelID, settings.Model, settings.OutgoingPromptTemplate, settings.HumanLanguage) {
			continue
		}

		slots[0].Function.Arguments = msg.ToolCalls[0].Function.Arguments
		for _, tc := range slots[1:] {
			tc.Function.Arguments = ""
		}

		changed = true
	}

	return changed
}

// persistTranslatedStream records the client-facing (translated) streamed turn
// alongside the original, pre-translation response_body persisted by
// persistRequestExecution, so both versions remain available for audit.
// Best-effort: failures are logged, never surfaced to the client.
func persistTranslatedStream(ctx context.Context, state *PersistenceState, chunks []*llm.Response) {
	if state == nil || state.RequestExec == nil {
		return
	}

	var last *llm.Response

	merged := map[int]*llm.Message{}
	var seq []int

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}

		last = chunk

		for _, choice := range chunk.Choices {
			if choice.Delta == nil {
				continue
			}

			msg, ok := merged[choice.Index]
			if !ok {
				msg = &llm.Message{Role: cohesiveRole(choice.Delta.Role)}
				merged[choice.Index] = msg
				seq = append(seq, choice.Index)
			}

			mergeDeltaIntoMessage(msg, choice.Delta)
		}
	}

	if last == nil || len(merged) == 0 {
		return
	}

	record := *last
	// The last chunk is often a terminal sentinel; the merged record carries real
	// content, so it must not inherit that marker.
	if record.Object == "" || record.Object == "[DONE]" {
		record.Object = "chat.completion.chunk"
	}

	record.Choices = make([]llm.Choice, 0, len(seq))

	for _, index := range seq {
		record.Choices = append(record.Choices, llm.Choice{
			Index:        index,
			Delta:        merged[index],
			FinishReason: lastFinishReason(chunks, index),
		})
	}

	persistTranslatedResponse(ctx, state, &record)
}

// cohesiveRole returns the role to assign to a merged message: the delta role if
// present, else the assistant default that streamed deltas implicitly carry.
func cohesiveRole(role string) string {
	if role == "" {
		return "assistant"
	}

	return role
}

// mergeDeltaIntoMessage appends one streamed delta onto the merged message,
// concatenating text, reasoning, and tool-call arguments, and preserving
// structured fields that only appear on some chunks. It is used to build the
// audit record of a translated streamed turn; it is deliberately not used to
// rebuild the client-facing stream (see translateOutgoingStreamChunks).
func mergeDeltaIntoMessage(msg *llm.Message, delta *llm.Message) {
	if delta.Content.Content != nil {
		if msg.Content.Content == nil {
			msg.Content.Content = lo.ToPtr(*delta.Content.Content)
		} else {
			*msg.Content.Content += *delta.Content.Content
		}
	}

	if len(delta.Content.MultipleContent) > 0 {
		msg.Content.MultipleContent = append(msg.Content.MultipleContent, delta.Content.MultipleContent...)
	}

	for _, tc := range delta.ToolCalls {
		mergeToolCall(msg, tc)
	}

	if delta.ReasoningContent != nil {
		if msg.ReasoningContent == nil {
			msg.ReasoningContent = lo.ToPtr(*delta.ReasoningContent)
		} else {
			*msg.ReasoningContent += *delta.ReasoningContent
		}
	}

	if delta.Reasoning != nil {
		if msg.Reasoning == nil {
			msg.Reasoning = lo.ToPtr(*delta.Reasoning)
		} else {
			*msg.Reasoning += *delta.Reasoning
		}
	}

	if delta.Refusal != "" {
		msg.Refusal += delta.Refusal
	}

	if len(delta.InlineToolResults) > 0 {
		msg.InlineToolResults = append(msg.InlineToolResults, delta.InlineToolResults...)
	}

	// Signature-bearing fields must survive the merge: dropping a reasoning
	// signature or encrypted content breaks multi-turn reasoning for providers
	// that require it (Anthropic thinking, Codex reasoning, ...). Take the last
	// non-empty value rather than concatenating, since these are opaque blobs.
	if delta.ReasoningSignature != nil && *delta.ReasoningSignature != "" {
		msg.ReasoningSignature = delta.ReasoningSignature
	}

	if delta.RedactedReasoningContent != nil && *delta.RedactedReasoningContent != "" {
		msg.RedactedReasoningContent = delta.RedactedReasoningContent
	}

	if len(delta.ReasoningItems) > 0 {
		msg.ReasoningItems = append(msg.ReasoningItems, delta.ReasoningItems...)
	}

	if delta.Name != nil && msg.Name == nil {
		msg.Name = delta.Name
	}

	if delta.CacheControl != nil {
		msg.CacheControl = delta.CacheControl
	}
}

// mergeToolCall merges one streamed tool-call delta into the merged message,
// matching by index, since a tool call's name/arguments stream across chunks.
// Both a Responses custom tool call's freeform script Input and a regular
// function call's JSON arguments accumulate; recording only one would leave the
// audit record missing the payload the agent actually runs.
func mergeToolCall(msg *llm.Message, delta llm.ToolCall) {
	for i := range msg.ToolCalls {
		if msg.ToolCalls[i].Index != delta.Index {
			continue
		}

		if delta.ID != "" {
			msg.ToolCalls[i].ID = delta.ID
		}

		if delta.Function.Name != "" {
			msg.ToolCalls[i].Function.Name = delta.Function.Name
		}

		msg.ToolCalls[i].Function.Arguments += delta.Function.Arguments

		if delta.ResponseCustomToolCall != nil {
			if msg.ToolCalls[i].ResponseCustomToolCall == nil {
				msg.ToolCalls[i].ResponseCustomToolCall = delta.ResponseCustomToolCall
			} else {
				msg.ToolCalls[i].ResponseCustomToolCall.Input += delta.ResponseCustomToolCall.Input
			}
		}

		return
	}

	msg.ToolCalls = append(msg.ToolCalls, delta)
}

// lastFinishReason returns the finish reason most recently reported for the
// given choice index, so the synthetic replacement chunk still terminates the
// stream the same way the original (buffered) stream would have.
func lastFinishReason(chunks []*llm.Response, index int) *string {
	for i := len(chunks) - 1; i >= 0; i-- {
		if chunks[i] == nil {
			continue
		}

		for _, choice := range chunks[i].Choices {
			if choice.Index == index && choice.FinishReason != nil {
				return choice.FinishReason
			}
		}
	}

	return nil
}

// persistTranslatedResponse records the client-facing (translated) response
// alongside the original response_body already persisted by
// persistRequestExecution, so both versions remain available for audit.
// Best-effort: failures are logged, never surfaced to the client.
func persistTranslatedResponse(ctx context.Context, state *PersistenceState, response *llm.Response) {
	if state == nil || state.RequestExec == nil {
		return
	}

	body, err := json.Marshal(response)
	if err != nil {
		log.Warn(ctx, "failed to marshal translated response for persistence", log.Cause(err))
		return
	}

	if err := state.RequestService.UpdateRequestExecutionTranslatedResponseBody(ctx, state.RequestExec.ID, objects.JSONRawMessage(body)); err != nil {
		log.Warn(ctx, "failed to persist translated response body", log.Cause(err))
	}
}

// interactiveToolNames are tools whose arguments are user-facing prose: an agent
// harness renders them as questions/options for the human to answer. Their text
// fields are translated toward the human language. Other tools are left alone --
// tool arguments are machine data (shell commands, file paths, code) that
// translating would corrupt, and their schema is the agent's, not ours.
var interactiveToolNames = map[string]struct{}{
	"request_user_input": {},
}

// interactiveToolTextFields are the argument keys inside an interactive tool's
// JSON arguments that hold user-facing prose. Nested under "questions"/"options".
var interactiveToolTextFields = map[string]struct{}{
	"header":      {},
	"question":    {},
	"label":       {},
	"description": {},
	"notes":       {},
}

// translateInteractiveToolCalls translates the prose fields inside the arguments
// of user-facing interactive tools in the message, leaving every other tool call
// untouched. Returns true when anything changed.
func translateInteractiveToolCalls(ctx context.Context, caller *translationCaller, msg *llm.Message, channelID int, model, promptTemplate, targetLanguage string) bool {
	changed := false

	for i := range msg.ToolCalls {
		call := &msg.ToolCalls[i]
		if _, ok := interactiveToolNames[strings.ToLower(call.Function.Name)]; !ok {
			continue
		}

		if strings.TrimSpace(call.Function.Arguments) == "" {
			continue
		}

		var args any
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			// Not JSON (or not the shape we expect): leave it untouched rather than
			// risk mangling a payload the harness must parse.
			continue
		}

		texts := collectInteractiveToolTexts(args)
		if len(texts) == 0 {
			continue
		}

		translatedTexts := make([]string, 0, len(texts))

		anyChanged := false

		for _, text := range texts {
			if !shouldTranslate(text, targetLanguage) {
				translatedTexts = append(translatedTexts, text)

				continue
			}

			translatedText, err := caller.translate(ctx, channelID, model, text, promptTemplate, targetLanguage)
			if err != nil {
				log.Warn(ctx, "failed to translate interactive tool text, keeping original", log.Cause(err))
				translatedTexts = append(translatedTexts, text)

				continue
			}

			if translatedText != text {
				anyChanged = true
			}

			translatedTexts = append(translatedTexts, translatedText)
		}

		if !anyChanged {
			continue
		}

		applyInteractiveToolTexts(args, translatedTexts, new(int))

		serialized, err := json.Marshal(args)
		if err != nil {
			log.Warn(ctx, "failed to re-marshal translated tool arguments, keeping original", log.Cause(err))
			continue
		}

		call.Function.Arguments = string(serialized)
		changed = true
	}

	return changed
}

// collectInteractiveToolTexts walks the tool arguments value and returns, in a
// stable order, every string that is the value of a key in
// interactiveToolTextFields. The order must match applyInteractiveToolTexts'
// consumption exactly, so both walk the same structure the same way.
func collectInteractiveToolTexts(value any) []string {
	var texts []string

	switch typed := value.(type) {
	case map[string]any:
		// Iterate keys in a deterministic order so collect/apply stay aligned.
		keys := sortedKeys(typed)

		for _, k := range keys {
			if _, ok := interactiveToolTextFields[k]; ok && isNonEmptyString(typed[k]) {
				texts = append(texts, typed[k].(string))

				continue
			}

			texts = append(texts, collectInteractiveToolTexts(typed[k])...)
		}
	case []any:
		for _, item := range typed {
			texts = append(texts, collectInteractiveToolTexts(item)...)
		}
	}

	return texts
}

// applyInteractiveToolTexts writes translated texts back into the arguments value
// in the same deterministic order collectInteractiveToolTexts produced them. Keys
// with empty values are not in the collected list, so they are skipped here too.
func applyInteractiveToolTexts(value any, texts []string, cursor *int) {
	switch typed := value.(type) {
	case map[string]any:
		keys := sortedKeys(typed)

		for _, k := range keys {
			if _, ok := interactiveToolTextFields[k]; ok && isNonEmptyString(typed[k]) {
				if *cursor < len(texts) {
					typed[k] = texts[*cursor]
					*cursor++
				}

				continue
			}

			applyInteractiveToolTexts(typed[k], texts, cursor)
		}
	case []any:
		for _, item := range typed {
			applyInteractiveToolTexts(item, texts, cursor)
		}
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Strings(keys)

	return keys
}

func isNonEmptyString(value any) bool {
	s, ok := value.(string)

	return ok && strings.TrimSpace(s) != ""
}

// translateMessageContent translates a message's text content in place,
// covering both the scalar Content and "text"-typed MultipleContent parts.
// All text segments in the message are translated in a single model call
// (see translateSegments) rather than one call per segment. Text segments that
// are structural rather than natural language are skipped (see shouldTranslate).
func translateMessageContent(ctx context.Context, caller *translationCaller, msg *llm.Message, channelID int, model, promptTemplate, targetLanguage string) error {
	// Collect the segments in message order: scalar Content first, then text parts.
	var (
		segments       []string
		segmentRefs    []*string
		contentSegment *string
		partSegments   []*string
	)

	if msg.Content.Content != nil {
		contentSegment = msg.Content.Content
	}

	for i := range msg.Content.MultipleContent {
		part := &msg.Content.MultipleContent[i]
		if !strings.EqualFold(part.Type, "text") || part.Text == nil {
			continue
		}

		partSegments = append(partSegments, part.Text)
	}

	// Map each candidate segment to its writable slot, keeping only the ones that
	// actually need translating.
	refs := append([]*string{}, contentSegment)
	refs = append(refs, partSegments...)

	for _, ref := range refs {
		if ref == nil || !shouldTranslate(*ref, targetLanguage) {
			continue
		}

		segments = append(segments, *ref)
		segmentRefs = append(segmentRefs, ref)
	}

	if len(segments) == 0 {
		return nil
	}

	translated, err := caller.translateSegments(ctx, channelID, model, segments, promptTemplate, targetLanguage)
	if err != nil {
		return err
	}

	for i, ref := range segmentRefs {
		*ref = translated[i]
	}

	return nil
}

// cjkRange reports whether r is a CJK ideograph or a fullwidth/ideographic
// punctuation mark that only appears in CJK text.
func cjkRange(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF, // CJK Unified Ideographs
		r >= 0x3400 && r <= 0x4DBF, // CJK Extension A
		r >= 0x3040 && r <= 0x30FF, // Hiragana / Katakana
		r >= 0xAC00 && r <= 0xD7AF, // Hangul syllables
		r >= 0x3000 && r <= 0x303F, // CJK symbols and punctuation
		r >= 0xFF00 && r <= 0xFFEF: // Fullwidth forms
		return true
	default:
		return false
	}
}

func containsCJK(s string) bool {
	for _, r := range s {
		if cjkRange(r) {
			return true
		}
	}

	return false
}

// looksLikeMarkupBlock reports whether the text is a single XML-like block, e.g.
// "<environment_context> ... </environment_context>". Whole-block harness context
// (environment, permissions, skills instructions) is structural data, not prose.
func looksLikeMarkupBlock(s string) bool {
	trimmed := strings.TrimSpace(s)
	if !strings.HasPrefix(trimmed, "<") {
		return false
	}

	// The first tag's name must also be the closing tag, so a block that merely
	// starts with markup but mixes in prose is not treated as structural.
	nameEnd := strings.IndexAny(trimmed, " >\t\n")
	if nameEnd <= 1 {
		return false
	}

	name := trimmed[1:nameEnd]
	if strings.ContainsAny(name, "/<") {
		return false
	}

	return strings.HasSuffix(trimmed, "</"+name+">")
}

// shouldTranslate reports whether a text segment is worth sending to the
// translation model. Segments with no target-opposite content are skipped to
// avoid the model answering with prose like "I don't see any text to translate"
// (which then gets written back as if it were the translation). Concretely:
//   - ASCII target languages: skip text with no CJK characters (it is already
//     English-plus-markup, so translating is a no-op at best);
//   - any target language: skip whole XML-like harness context blocks, which are
//     structural data even when they contain no translatable prose.
//
// The CJK heuristic is deliberately one-sided: it only recognizes "nothing to do
// for an ASCII target". It never claims to know a text is already Chinese, so it
// cannot skip text that a Chinese target still needs.
func shouldTranslate(text, targetLanguage string) bool {
	if strings.TrimSpace(text) == "" {
		return false
	}

	if looksLikeMarkupBlock(text) {
		return false
	}

	if isAsciiLanguage(targetLanguage) && !containsCJK(text) {
		return false
	}

	return true
}

// isAsciiLanguage reports whether the target language name refers to a
// Latin/ASCII-script language. Only used to gate the "already ASCII" skip.
func isAsciiLanguage(language string) bool {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "english", "en", "spanish", "es", "french", "fr", "german", "de",
		"portuguese", "pt", "italian", "it", "dutch", "nl", "swedish", "sv",
		"norwegian", "no", "danish", "da", "finnish", "fi", "polish", "pl",
		"turkish", "tr", "indonesian", "id", "malay", "ms", "vietnamese", "vi",
		"tagalog", "filipino", "czech", "cs", "hungarian", "hu", "romanian", "ro":
		return true
	default:
		return false
	}
}

// structuredOutputUserPromptMarker is the label an agent harness inserts before
// the user's own text in a structured-output request. Only that trailing span is
// user content; everything before it is harness instruction.
const structuredOutputUserPromptMarker = "User prompt:"

// translateStructuredOutputTail translates the user text embedded after
// structuredOutputUserPromptMarker in a structured-output request's last scoped
// message, leaving the surrounding harness instructions untouched. Returns true
// when the text actually changed. Best-effort: any failure leaves the request
// unmodified so the harness job still runs on the original text.
func translateStructuredOutputTail(ctx context.Context, caller *translationCaller, request *llm.Request, settings *biz.TranslationSettings) bool {
	index := lastTranslatableMessageIndex(request.Messages)
	if index < 0 {
		return false
	}

	msg := &request.Messages[index]

	translateTail := func(segment *string) (string, bool) {
		if segment == nil {
			return "", false
		}

		markerIndex := strings.LastIndex(*segment, structuredOutputUserPromptMarker)
		if markerIndex < 0 {
			return "", false
		}

		head := (*segment)[:markerIndex+len(structuredOutputUserPromptMarker)]
		tail := (*segment)[markerIndex+len(structuredOutputUserPromptMarker):]

		if strings.TrimSpace(tail) == "" {
			return "", false
		}

		if !shouldTranslate(tail, settings.AgentLanguage) {
			return "", false
		}

		translated, err := caller.translateWithInstructionCached(ctx, settings.ChannelID, settings.Model, tail, settings.IncomingPromptTemplate, settings.AgentLanguage, "")
		if err != nil {
			log.Warn(ctx, "failed to translate structured-output user prompt, keeping original", log.Cause(err))
			return "", false
		}

		if translated == tail {
			return "", false
		}

		*segment = head + translated

		return *segment, true
	}

	if msg.Content.Content != nil {
		if updated, changed := translateTail(msg.Content.Content); changed {
			msg.Content.Content = &updated

			return true
		}
	}

	changed := false
	for i := range msg.Content.MultipleContent {
		part := &msg.Content.MultipleContent[i]
		if !strings.EqualFold(part.Type, "text") {
			continue
		}

		if _, partChanged := translateTail(part.Text); partChanged {
			changed = true
		}
	}

	return changed
}

// hasStructuredOutputConstraint reports whether the request forces a structured
// response format (JSON object / JSON schema). Such requests are agent-harness
// internal jobs (e.g. generating a task title), not free-form conversation, and
// translating either direction is meaningless or harmful. "text" is the normal
// prose format and is not a constraint.
func hasStructuredOutputConstraint(request *llm.Request) bool {
	if request == nil || request.ResponseFormat == nil {
		return false
	}

	return request.ResponseFormat.Type != "" && request.ResponseFormat.Type != "text"
}

// lastTranslatableMessageIndex returns the index of the most recent translatable
// message, or -1 if none match.
func lastTranslatableMessageIndex(messages []llm.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if translationRoleMatches(messages[i].Role) {
			return i
		}
	}

	return -1
}

// translationRoles are the message roles automatic translation applies to. This
// is a correctness constraint, not a preference: user messages carry the human
// language and assistant messages the agent's, so both must be normalized or the
// model sees a mismatched history. Other roles are excluded on purpose --
// system/developer are the client's own (English) harness instructions, and tool
// messages are machine data (command output, file contents, paths) that
// translating would corrupt.
var translationRoles = map[string]struct{}{
	"user":      {},
	"assistant": {},
}

// translationRoleMatches reports whether role is translated.
func translationRoleMatches(role string) bool {
	_, ok := translationRoles[strings.ToLower(role)]

	return ok
}
