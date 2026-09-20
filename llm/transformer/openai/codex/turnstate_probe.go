package codex

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer/openai/codex/turnstate"
)

const (
	// defaultTurnStateProbeInterval is how often the probe looks for a bucket
	// that is missing a template or close to losing one.
	defaultTurnStateProbeInterval = time.Minute

	// turnStateProbeRefreshMargin is the remaining lifetime below which a
	// template is re-harvested. The signed window is one hour.
	turnStateProbeRefreshMargin = 5 * time.Minute

	// turnStateProbeTimeout bounds one upstream call. Only the response
	// headers are wanted and they arrive before the SSE body, so this is
	// generous.
	turnStateProbeTimeout = 60 * time.Second

	// turnStateProbePauseOnRefusal is how long the probe backs off after the
	// credential itself is refused (401/403) or rate limited (429). Continuing
	// to fire is what turns a throttled account into a rate limited one.
	turnStateProbePauseOnRefusal = 10 * time.Minute

	// turnStateProbeUserAgent mirrors a real codex-tui build so the harvest
	// looks like ordinary Codex traffic.
	turnStateProbeUserAgent = "codex-tui/0.154.0 (Ubuntu 24.04; x86_64) (codex-tui; 0.154.0)"
)

// TurnStateProbeConfig describes one channel's harvest loop.
type TurnStateProbeConfig struct {
	// Models are the upstream model names to keep warm.
	Models []string

	// Client performs the harvest request. It should be the channel's own
	// client so the probe leaves through the same proxy as business traffic.
	Client *http.Client

	// BaseURL is the official Codex base URL, for example
	// https://chatgpt.com/backend-api/codex#.
	BaseURL string

	// Store is the destination store. Nil uses the process wide store.
	Store *turnstate.Store

	// Interval overrides the default probe interval. Mainly for tests.
	Interval time.Duration
}

// TurnStateProbe keeps the turn-state store warm for one Codex credential.
//
// It fires at most one upstream request per interval, only for a bucket that
// is missing a template or close to expiring, so a healthy channel settles
// into roughly one request per model per hour. Each request is a minimal
// "ping" whose SSE body is dropped as soon as the headers are read.
type TurnStateProbe struct {
	config TurnStateProbeConfig
	tokens oauth.TokenGetter

	started  atomic.Bool
	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
	cancel   context.CancelFunc

	pauseMu    sync.Mutex
	pauseUntil time.Time
}

// NewTurnStateProbe builds a probe, or returns nil when the configuration
// cannot support one.
func NewTurnStateProbe(config TurnStateProbeConfig, tokens oauth.TokenGetter) *TurnStateProbe {
	if config.Client == nil || tokens == nil || !turnstate.Enabled() {
		return nil
	}

	models := make([]string, 0, len(config.Models))
	for _, model := range config.Models {
		model = strings.TrimSpace(model)
		if model == "" || slices.Contains(models, model) {
			continue
		}
		models = append(models, model)
	}
	if len(models) == 0 {
		return nil
	}
	config.Models = models

	if config.Store == nil {
		config.Store = turnstate.Default()
	}
	if config.Interval <= 0 {
		config.Interval = defaultTurnStateProbeInterval
	}
	if strings.TrimSpace(config.BaseURL) == "" {
		config.BaseURL = codexBaseURL
	}

	return &TurnStateProbe{
		config: config,
		tokens: tokens,
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

// Start launches the harvest loop. The first pass runs immediately so a
// channel that has never been used still gets a template.
func (p *TurnStateProbe) Start() {
	if p == nil || !p.started.CompareAndSwap(false, true) {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	p.cancel = cancel

	go p.run(ctx)
}

// Stop ends the harvest loop and waits for the goroutine to exit. It is safe
// to call more than once.
func (p *TurnStateProbe) Stop() {
	if p == nil || !p.started.Load() {
		return
	}

	p.stopOnce.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		close(p.stop)
	})

	<-p.done
}

func (p *TurnStateProbe) run(ctx context.Context) {
	defer close(p.done)
	defer func() {
		if r := recover(); r != nil {
			slog.Error("codex turn-state probe panicked", slog.Any("recover", r))
		}
	}()

	p.harvest(ctx)

	ticker := time.NewTicker(p.config.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-p.stop:
			return
		case <-ticker.C:
			p.harvest(ctx)
		}
	}
}

// harvest refreshes at most one bucket per call. Serialising the work keeps a
// multi model channel from firing several upstream requests at once, which is
// what provokes the upstream rate limit.
func (p *TurnStateProbe) harvest(ctx context.Context) {
	if p == nil {
		return
	}

	p.pauseMu.Lock()
	paused := time.Now().Before(p.pauseUntil)
	p.pauseMu.Unlock()
	if paused {
		return
	}

	creds, err := p.tokens.Get(ctx)
	if err != nil || creds == nil || strings.TrimSpace(creds.AccessToken) == "" {
		return
	}

	accountID := ExtractChatGPTAccountIDFromJWT(creds.AccessToken)
	if accountID == "" {
		return
	}

	for _, model := range p.config.Models {
		if ctx.Err() != nil {
			return
		}
		if p.config.Store.Fresh(accountID, model, turnStateProbeRefreshMargin) {
			continue
		}

		p.fire(ctx, creds.AccessToken, accountID, model)

		return
	}
}

// fire makes one harvest request and stores what comes back. Only the
// response headers matter, so the body is closed without being consumed.
func (p *TurnStateProbe) fire(ctx context.Context, accessToken, accountID, model string) {
	payload, err := json.Marshal(map[string]any{
		"model":  model,
		"stream": true,
		"store":  false,
		"input": []map[string]any{{
			"type": "message",
			"role": "user",
			"content": []map[string]any{{
				"type": "input_text",
				"text": "ping",
			}},
		}},
		"reasoning":           map[string]any{"effort": "low"},
		"tool_choice":         "auto",
		"parallel_tool_calls": false,
	})
	if err != nil {
		return
	}

	callCtx, cancel := context.WithTimeout(ctx, turnStateProbeTimeout)
	defer cancel()

	url := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSpace(p.config.BaseURL), "#"), "/") + "/responses"
	request, err := http.NewRequestWithContext(callCtx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return
	}

	request.Header.Set("Authorization", "Bearer "+accessToken)
	request.Header.Set("Chatgpt-Account-Id", accountID)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	request.Header.Set("Originator", "codex-tui")
	request.Header.Set("Session-Id", uuid.NewString())
	request.Header.Set("User-Agent", turnStateProbeUserAgent)

	response, err := p.config.Client.Do(request)
	if err != nil {
		slog.DebugContext(ctx, "codex turn-state probe failed", slog.String("error", err.Error()))

		return
	}
	defer func() {
		_ = response.Body.Close()
	}()

	switch response.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusTooManyRequests:
		slog.WarnContext(ctx, "codex turn-state probe paused after upstream refusal",
			slog.Int("status", response.StatusCode),
			slog.Duration("pause", turnStateProbePauseOnRefusal))
		p.pauseMu.Lock()
		p.pauseUntil = time.Now().Add(turnStateProbePauseOnRefusal)
		p.pauseMu.Unlock()

		return
	}

	if response.StatusCode != http.StatusOK {
		return
	}

	value := strings.TrimSpace(response.Header.Get(turnstate.Header))
	if value == "" {
		return
	}

	if p.config.Store.Observe(accountID, model, value) {
		slog.DebugContext(ctx, "codex turn-state harvested",
			slog.String("model", model),
			slog.Int("length", len(value)))
	}
}
