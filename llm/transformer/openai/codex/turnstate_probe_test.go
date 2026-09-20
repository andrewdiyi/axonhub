package codex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer/openai/codex/turnstate"
)

type probeTokenGetter struct {
	creds *oauth.OAuthCredentials
}

func (g probeTokenGetter) Get(_ context.Context) (*oauth.OAuthCredentials, error) {
	return g.creds, nil
}

func TestTurnStateProbeHarvestsTemplate(t *testing.T) {
	token := testAccessTokenWithAccountID(t)
	template := testTurnStateToken(t, time.Now(), 217)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer "+token, r.Header.Get("Authorization"))
		assert.Equal(t, testChatAccountID, r.Header.Get("Chatgpt-Account-Id"))

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		assert.True(t, strings.Contains(string(body), `"gpt-6-astra"`), "probe body must name the model")

		w.Header().Set(turnstate.Header, template)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := turnstate.NewStore()
	probe := NewTurnStateProbe(TurnStateProbeConfig{
		Models:   []string{"gpt-6-astra", "gpt-6-astra", "  "},
		Client:   server.Client(),
		BaseURL:  server.URL,
		Store:    store,
		Interval: 20 * time.Millisecond,
	}, probeTokenGetter{creds: &oauth.OAuthCredentials{AccessToken: token}})
	require.NotNil(t, probe)

	probe.Start()
	defer probe.Stop()

	require.Eventually(t, func() bool {
		entry, ok := store.Good(testChatAccountID, "gpt-6-astra")

		return ok && entry.Value == template
	}, 3*time.Second, 10*time.Millisecond)
}

func TestTurnStateProbeSkipsFreshTemplate(t *testing.T) {
	token := testAccessTokenWithAccountID(t)
	template := testTurnStateToken(t, time.Now(), 217)

	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.Header().Set(turnstate.Header, template)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := turnstate.NewStore()
	require.True(t, store.Observe(testChatAccountID, "gpt-6-astra", template))

	probe := NewTurnStateProbe(TurnStateProbeConfig{
		Models:   []string{"gpt-6-astra"},
		Client:   server.Client(),
		BaseURL:  server.URL,
		Store:    store,
		Interval: 20 * time.Millisecond,
	}, probeTokenGetter{creds: &oauth.OAuthCredentials{AccessToken: token}})
	require.NotNil(t, probe)

	probe.Start()
	time.Sleep(150 * time.Millisecond)
	probe.Stop()

	assert.Zero(t, calls, "a template with a full hour of life left must not be re-harvested")
}

func TestTurnStateProbePausesAfterRateLimit(t *testing.T) {
	token := testAccessTokenWithAccountID(t)

	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer server.Close()

	store := turnstate.NewStore()
	probe := NewTurnStateProbe(TurnStateProbeConfig{
		Models:   []string{"gpt-6-astra"},
		Client:   server.Client(),
		BaseURL:  server.URL,
		Store:    store,
		Interval: 10 * time.Millisecond,
	}, probeTokenGetter{creds: &oauth.OAuthCredentials{AccessToken: token}})
	require.NotNil(t, probe)

	probe.Start()
	time.Sleep(120 * time.Millisecond)
	probe.Stop()

	assert.Equal(t, 1, calls, "the probe must back off after the upstream refuses the credential")
}
