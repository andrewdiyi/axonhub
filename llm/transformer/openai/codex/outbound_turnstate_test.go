package codex

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"net/http"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer/openai/codex/turnstate"
)

func testTurnStateToken(t *testing.T, issuedAt time.Time, size int) string {
	t.Helper()

	raw := make([]byte, size)
	raw[0] = 0x80
	binary.BigEndian.PutUint64(raw[1:9], uint64(issuedAt.Unix()))

	return base64.URLEncoding.EncodeToString(raw)
}

func newTurnStateOutbound(t *testing.T, baseURL string, store *turnstate.Store) *OutboundTransformer {
	t.Helper()

	outbound, err := NewOutboundTransformer(Params{
		BaseURL: baseURL,
		TokenProvider: staticTokenGetter{
			creds: &oauth.OAuthCredentials{
				AccessToken: testAccessTokenWithAccountID(t),
				ExpiresAt:   time.Now().Add(time.Hour),
			},
		},
		TurnState:        store,
		TurnStateEnabled: true,
	})
	require.NoError(t, err)

	return outbound
}

func newTurnStateRequest(model string) *llm.Request {
	return &llm.Request{
		Model: model,
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
		Stream: lo.ToPtr(true),
	}
}

func TestCodexOutbound_InjectsTurnStateAndHarvestsResponse(t *testing.T) {
	ctx := context.Background()
	store := turnstate.NewStore()
	now := time.Now()
	stored := testTurnStateToken(t, now.Add(-time.Minute), 217)

	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", stored))

	outbound := newTurnStateOutbound(t, "https://chatgpt.com/backend-api/codex#", store)

	hreq, err := outbound.TransformRequest(ctx, newTurnStateRequest("gpt-6-astra"))
	require.NoError(t, err)
	assert.Equal(t, stored, hreq.Headers.Get(TurnStateHeader), "the stored template must be injected")
	require.NotNil(t, hreq.ResponseHeaderHook, "a harvest hook must be installed")

	// The upstream mints a fresh value; the hook must store it under the same
	// account and model bucket.
	fresh := testTurnStateToken(t, now, 249)
	hreq.ResponseHeaderHook(http.Header{TurnStateHeader: []string{fresh}})

	entry, ok := store.Good(testChatAccountID, "gpt-6-astra")
	require.True(t, ok)
	assert.Equal(t, fresh, entry.Value)
}

func TestCodexOutbound_TurnStateStaysWithinBucket(t *testing.T) {
	ctx := context.Background()
	store := turnstate.NewStore()
	now := time.Now()
	stored := testTurnStateToken(t, now.Add(-time.Minute), 217)

	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", stored))

	outbound := newTurnStateOutbound(t, "https://chatgpt.com/backend-api/codex#", store)

	// A different model never receives another model's template.
	hreq, err := outbound.TransformRequest(ctx, newTurnStateRequest("gpt-5.6-sol"))
	require.NoError(t, err)
	assert.Empty(t, hreq.Headers.Get(TurnStateHeader))

	// A degraded value in the store's place is not enough to inject.
	store.Forget(testChatAccountID, "gpt-6-astra")
	assert.False(t, store.Observe(testChatAccountID, "gpt-6-astra", testTurnStateToken(t, now, 233)))

	hreq, err = outbound.TransformRequest(ctx, newTurnStateRequest("gpt-6-astra"))
	require.NoError(t, err)
	assert.Empty(t, hreq.Headers.Get(TurnStateHeader))
}

func TestCodexOutbound_RelayNeverTouchesTurnState(t *testing.T) {
	ctx := context.Background()
	store := turnstate.NewStore()
	now := time.Now()
	stored := testTurnStateToken(t, now.Add(-time.Minute), 217)

	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", stored))

	outbound := newTurnStateOutbound(t, "https://relay.example.com/v1", store)

	hreq, err := outbound.TransformRequest(ctx, newTurnStateRequest("gpt-6-astra"))
	require.NoError(t, err)
	assert.Empty(t, hreq.Headers.Get(TurnStateHeader))
	assert.Nil(t, hreq.ResponseHeaderHook)
}

func TestCodexOutbound_CompactSkipsTurnState(t *testing.T) {
	ctx := context.Background()
	store := turnstate.NewStore()
	now := time.Now()

	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", testTurnStateToken(t, now.Add(-time.Minute), 217)))

	outbound := newTurnStateOutbound(t, "https://chatgpt.com/backend-api/codex#", store)

	req := newTurnStateRequest("gpt-6-astra")
	req.RequestType = llm.RequestTypeCompact
	req.Compact = &llm.CompactRequest{Input: req.Messages}

	hreq, err := outbound.TransformRequest(ctx, req)
	require.NoError(t, err)
	assert.Empty(t, hreq.Headers.Get(TurnStateHeader))
	assert.Nil(t, hreq.ResponseHeaderHook)
}

func TestCodexOutbound_TurnStateDisabledByChannelSetting(t *testing.T) {
	ctx := context.Background()
	store := turnstate.NewStore()
	now := time.Now()

	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", testTurnStateToken(t, now.Add(-time.Minute), 217)))

	outbound, err := NewOutboundTransformer(Params{
		BaseURL: "https://chatgpt.com/backend-api/codex#",
		TokenProvider: staticTokenGetter{
			creds: &oauth.OAuthCredentials{
				AccessToken: testAccessTokenWithAccountID(t),
				ExpiresAt:   time.Now().Add(time.Hour),
			},
		},
		TurnState:        store,
		TurnStateEnabled: false,
	})
	require.NoError(t, err)

	hreq, err := outbound.TransformRequest(ctx, newTurnStateRequest("gpt-6-astra"))
	require.NoError(t, err)
	assert.Empty(t, hreq.Headers.Get(TurnStateHeader), "a disabled channel must not inject")
	assert.Nil(t, hreq.ResponseHeaderHook, "a disabled channel must not harvest")
}
