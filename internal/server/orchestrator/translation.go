package orchestrator

import (
	"context"
	"encoding/json"
	"strings"

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

		if err := translateMessageContent(ctx, caller, msg, settings.ChannelID, settings.Model, settings.OutgoingPromptTemplate, settings.HumanLanguage); err != nil {
			log.Warn(ctx, "failed to translate outgoing response, passing through original text", log.Cause(err))
			continue
		}

		translated = true
	}

	if translated {
		persistTranslatedResponse(ctx, state, response)
	}

	return response, nil
}

// OnOutboundLlmStream buffers the full streamed reply, translates it once
// (there is no hook that exposes the full text of a streamed turn before
// delivery, so this is the simplest correct behavior), then emits it as a
// single replacement chunk preserving the final chunk's finish reason and
// usage. This trades incremental streaming UX for correctness: the client
// receives nothing until generation and translation both complete.
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

	texts := map[int]*strings.Builder{}

	var last *llm.Response

	// hasNonTextContent reports whether any chunk carries tool calls, inline
	// tool results, or reasoning. The synthetic single-chunk replacement built
	// below only has a Role and translated Content; it cannot represent these
	// fields without a full chunk-merge, so a turn that includes any of them
	// is passed through unmodified instead of being collapsed and silently
	// losing that data (e.g. a tool call turning into an empty response).
	hasNonTextContent := false

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}

		last = chunk

		for _, choice := range chunk.Choices {
			if choice.Delta == nil {
				continue
			}

			if len(choice.Delta.ToolCalls) > 0 || len(choice.Delta.InlineToolResults) > 0 ||
				choice.Delta.Refusal != "" ||
				(choice.Delta.ReasoningContent != nil && *choice.Delta.ReasoningContent != "") ||
				(choice.Delta.Reasoning != nil && *choice.Delta.Reasoning != "") {
				hasNonTextContent = true
			}

			// Delta.Role is typically only set on the first chunk of a choice;
			// later chunks implicitly continue the same (assistant) message.
			role := choice.Delta.Role
			if role == "" {
				role = "assistant"
			}

			if !translationRoleMatches(role) {
				continue
			}

			builder, ok := texts[choice.Index]
			if !ok {
				builder = &strings.Builder{}
				texts[choice.Index] = builder
			}

			if choice.Delta.Content.Content != nil {
				builder.WriteString(*choice.Delta.Content.Content)
			}

			for _, part := range choice.Delta.Content.MultipleContent {
				if strings.EqualFold(part.Type, "text") && part.Text != nil {
					builder.WriteString(*part.Text)
				}
			}
		}
	}

	if hasNonTextContent || last == nil {
		return streams.SliceStream(chunks), nil
	}

	// Drop choices that matched scope but never accumulated real text (e.g. a
	// role-only chunk with no content), rather than translating an empty
	// string into an equally-empty "translated" chunk.
	for index, builder := range texts {
		if strings.TrimSpace(builder.String()) == "" {
			delete(texts, index)
		}
	}

	if len(texts) == 0 {
		return streams.SliceStream(chunks), nil
	}

	final := *last
	// last is frequently a terminal/sentinel event (e.g. Object == "[DONE]"),
	// which the pipeline's empty-response detector (hasResponseContent) checks
	// *before* it even looks at Choices. Clear that inherited marker: final is
	// about to carry real (translated or fail-open) text, not a sentinel, so
	// leaving it in place would make a perfectly good response look empty and
	// trigger spurious retries.
	final.Object = "chat.completion.chunk"
	final.Choices = make([]llm.Choice, 0, len(texts))

	translated := false

	for index, builder := range texts {
		text := builder.String()

		translatedText, err := caller.translate(ctx, settings.ChannelID, settings.Model, text, settings.OutgoingPromptTemplate, settings.HumanLanguage)
		if err != nil {
			log.Warn(ctx, "failed to translate outgoing streamed response, passing through original text", log.Cause(err))
			translatedText = text
		} else {
			translated = true
		}

		final.Choices = append(final.Choices, llm.Choice{
			Index: index,
			Delta: &llm.Message{
				Role:    "assistant",
				Content: llm.MessageContent{Content: &translatedText},
			},
			FinishReason: lastFinishReason(chunks, index),
		})
	}

	if translated {
		persistTranslatedResponse(ctx, state, &final)
	}

	return streams.SliceStream([]*llm.Response{&final}), nil
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
