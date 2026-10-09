package orchestrator

import (
	"context"
	"encoding/json"
	"slices"
	"strings"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/internal/objects"
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

// OnInboundLlmRequest translates the single most recent message matching the
// configured scope, toward AgentLanguage, before the request reaches channel
// selection/the upstream provider. Only the most recent message is
// translated (not the whole resent history) to avoid re-translating every
// prior turn on every new message, which would multiply internal translation
// calls as a conversation grows.
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

	index := lastScopedMessageIndex(request.Messages, settings.Scopes)
	if index < 0 {
		return request, nil
	}

	if err := translateMessageContent(ctx, caller, &request.Messages[index], settings.ChannelID, settings.Model, settings.IncomingPromptTemplate, settings.AgentLanguage); err != nil {
		log.Warn(ctx, "failed to translate incoming request, passing through original text", log.Cause(err))
		return request, nil
	}

	return request, nil
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

		if !translationScopeMatches(settings.Scopes, role) {
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

	chunks, err := streams.All(stream)
	if err != nil {
		return streams.SliceStream(chunks), err
	}

	texts := map[int]*strings.Builder{}

	var last *llm.Response

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}

		last = chunk

		for _, choice := range chunk.Choices {
			if choice.Delta == nil {
				continue
			}

			// Delta.Role is typically only set on the first chunk of a choice;
			// later chunks implicitly continue the same (assistant) message.
			role := choice.Delta.Role
			if role == "" {
				role = "assistant"
			}

			if !translationScopeMatches(settings.Scopes, role) {
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

	if len(texts) == 0 || last == nil {
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
func translateMessageContent(ctx context.Context, caller *translationCaller, msg *llm.Message, channelID int, model, promptTemplate, targetLanguage string) error {
	if msg.Content.Content != nil {
		translated, err := caller.translate(ctx, channelID, model, *msg.Content.Content, promptTemplate, targetLanguage)
		if err != nil {
			return err
		}

		msg.Content.Content = &translated
	}

	for i := range msg.Content.MultipleContent {
		part := &msg.Content.MultipleContent[i]
		if !strings.EqualFold(part.Type, "text") || part.Text == nil {
			continue
		}

		translated, err := caller.translate(ctx, channelID, model, *part.Text, promptTemplate, targetLanguage)
		if err != nil {
			return err
		}

		part.Text = &translated
	}

	return nil
}

// lastScopedMessageIndex returns the index of the most recent message whose
// role is in scopes, or -1 if none match.
func lastScopedMessageIndex(messages []llm.Message, scopes []objects.TranslationScope) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if translationScopeMatches(scopes, messages[i].Role) {
			return i
		}
	}

	return -1
}

// translationScopeMatches reports whether role is configured for translation.
// Unlike prompt-protection's scope check, an empty scope list matches nothing:
// translation is an explicit opt-in per role, not a default-allow regex gate.
func translationScopeMatches(scopes []objects.TranslationScope, role string) bool {
	if len(scopes) == 0 {
		return false
	}

	return slices.Contains(scopes, objects.TranslationScope(strings.ToLower(role)))
}
