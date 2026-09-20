// Package turnstate stores the X-Codex-Turn-State values the official Codex
// backend mints and decides which value an outbound request should carry.
//
// The value is a Fernet token: a 0x80 version byte followed by an eight byte
// big-endian Unix timestamp (the moment the upstream issued it) and the
// encrypted payload. The upstream enforces a one hour validity window measured
// from that embedded timestamp, so expiry is computed from the token itself
// rather than from the moment this process happened to observe it.
//
// Values are classified by decoded size rather than by string length, so a
// value survives base64 padding being added or stripped in transit. Observed
// decoded sizes:
//
//	217 / 249 bytes  reusable "good" templates (individual / Team plans)
//	233 / 267 bytes  degraded counterparts minted on the throttled path
//
// The degraded form carries exactly one extra AES block (16 bytes) over its
// individual counterpart, which is why the byte sizes, not the encrypted
// contents, are the reliable tell.
//
// Only good values are stored. A degraded value is never kept: the point of
// the store is to replace it with a fresh good value for the same bucket.
//
// A value may not be reused across accounts, and may not be reused across
// models even within one account. Both boundaries are enforced by the Bucket
// key; nothing in this package crosses them.
package turnstate

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"time"
)

// Header is the upstream header that carries the turn-state value.
const Header = "X-Codex-Turn-State"

// DisableEnv is the environment variable that turns turn-state handling off
// without a rebuild. Any of off/0/false/no disables it; anything else, or an
// unset variable, keeps it enabled.
const DisableEnv = "AXONHUB_CODEX_TURN_STATE"

// Enabled reports whether turn-state injection and harvesting are active.
func Enabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(DisableEnv))) {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

// DefaultTTL mirrors the one hour window the upstream signs into the token.
const DefaultTTL = time.Hour

// clockSkew tolerates a token minted a little in the future, which can happen
// when the upstream clock runs ahead of ours. Anything further ahead is not
// decodable as a sane issue time and is rejected.
const clockSkew = 2 * time.Minute

// Decoded size sets observed in the wild. Kept as slices instead of two pairs
// so a new plan shape can be added without touching the classification logic.
var (
	goodSizes     = []int{217, 249}
	degradedSizes = []int{233, 267}
)

// Kind classifies a decodeable turn-state value.
type Kind uint8

const (
	// KindUnknown is a value that is not a recognised Fernet token shape.
	KindUnknown Kind = iota
	// KindGood is a reusable template length.
	KindGood
	// KindDegraded is the throttled path length, replaced when possible.
	KindDegraded
)

func (k Kind) String() string {
	switch k {
	case KindGood:
		return "good"
	case KindDegraded:
		return "degraded"
	default:
		return "unknown"
	}
}

// Bucket is the isolation boundary of a stored value: one account and one
// model. The upstream rejects a value used outside its bucket.
type Bucket struct {
	AccountID string
	Model     string
}

// Entry is one stored template.
type Entry struct {
	Bucket
	Value       string
	Kind        Kind
	IssuedAt    time.Time
	HarvestedAt time.Time
}

// ExpiresAt reports when the upstream window of the entry closes.
func (e Entry) ExpiresAt(ttl time.Duration) time.Time {
	if ttl <= 0 {
		ttl = DefaultTTL
	}

	return e.IssuedAt.Add(ttl)
}

// live reports whether the entry is usable at now: issued in the past (within
// the clock skew allowance) and not past its signed window.
func (e Entry) live(now time.Time, ttl time.Duration) bool {
	if e.Value == "" {
		return false
	}
	if e.IssuedAt.After(now.Add(clockSkew)) {
		return false
	}

	return now.Before(e.ExpiresAt(ttl))
}

// decode unwraps the Fernet framing shared by every value: base64url (padding
// optional), a 0x80 version byte and an eight byte big-endian timestamp.
func decode(value string) ([]byte, time.Time, bool) {
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(strings.TrimSpace(value), "="))
	if err != nil || len(raw) < 9 || raw[0] != 0x80 {
		return nil, time.Time{}, false
	}

	secs := binary.BigEndian.Uint64(raw[1:9])
	if secs > math.MaxInt64 {
		return nil, time.Time{}, false
	}

	return raw, time.Unix(int64(secs), 0).UTC(), true
}

// FernetIssuedAt extracts the issuance time embedded in a turn-state value.
// The second return value is false when the value is not a decodable Fernet
// token, in which case callers must not treat it as usable.
func FernetIssuedAt(value string) (time.Time, bool) {
	_, issuedAt, ok := decode(value)

	return issuedAt, ok
}

// Classify reports the kind of a value plus its embedded issuance time. The
// bool is false for values that are not a recognised Fernet token shape, which
// includes every value whose length this package does not know.
func Classify(value string) (Kind, time.Time, bool) {
	raw, issuedAt, ok := decode(value)
	if !ok {
		return KindUnknown, time.Time{}, false
	}

	switch {
	case slices.Contains(goodSizes, len(raw)):
		return KindGood, issuedAt, true
	case slices.Contains(degradedSizes, len(raw)):
		return KindDegraded, issuedAt, true
	default:
		return KindUnknown, issuedAt, false
	}
}

// Option customises a Store. Intended for tests and for callers that want a
// different window than the upstream default.
type Option func(*Store)

// WithTTL overrides the validity window used for stored entries.
func WithTTL(ttl time.Duration) Option {
	return func(s *Store) {
		if ttl > 0 {
			s.ttl = ttl
		}
	}
}

// WithClock overrides the clock, so tests do not depend on wall time.
func WithClock(now func() time.Time) Option {
	return func(s *Store) {
		if now != nil {
			s.now = now
		}
	}
}

// Store keeps one live template per bucket. It is safe for concurrent use.
type Store struct {
	mu      sync.Mutex
	ttl     time.Duration
	now     func() time.Time
	buckets map[Bucket]Entry
}

// NewStore creates an empty store.
func NewStore(opts ...Option) *Store {
	store := &Store{
		ttl:     DefaultTTL,
		now:     time.Now,
		buckets: map[Bucket]Entry{},
	}
	for _, opt := range opts {
		opt(store)
	}

	return store
}

// DefaultStore is the process wide store. Every Codex outbound created by the
// server shares it, which is what keeps a state harvested on one request
// usable by the next one.
var defaultStore = NewStore()

// Default returns the process wide store.
func Default() *Store {
	return defaultStore
}

// Observe stores value as the template for its bucket when it is a fresh,
// reusable (good) value. Degraded and unrecognised values are ignored: they
// are never installed as a template. The bool reports whether the store now
// holds a template at least as new as the observed one.
func (s *Store) Observe(accountID, model, value string) bool {
	if s == nil {
		return false
	}

	accountID = strings.TrimSpace(accountID)
	model = strings.TrimSpace(model)
	if accountID == "" || model == "" {
		return false
	}

	kind, issuedAt, ok := Classify(value)
	if !ok || kind != KindGood {
		return false
	}

	now := s.now()
	if issuedAt.After(now.Add(clockSkew)) {
		return false
	}

	key := Bucket{AccountID: accountID, Model: model}
	entry := Entry{
		Bucket:      key,
		Value:       strings.TrimSpace(value),
		Kind:        kind,
		IssuedAt:    issuedAt,
		HarvestedAt: now,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.purgeLocked(now)

	if existing, found := s.buckets[key]; found && existing.IssuedAt.After(issuedAt) {
		// A late response carrying an older token must not age the bucket
		// backwards.
		return true
	}

	s.buckets[key] = entry

	return true
}

// Good returns the live good template for a bucket, if any.
func (s *Store) Good(accountID, model string) (Entry, bool) {
	if s == nil {
		return Entry{}, false
	}

	key := Bucket{
		AccountID: strings.TrimSpace(accountID),
		Model:     strings.TrimSpace(model),
	}
	if key.AccountID == "" || key.Model == "" {
		return Entry{}, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, found := s.buckets[key]
	if !found {
		return Entry{}, false
	}

	if !entry.live(s.now(), s.ttl) {
		delete(s.buckets, key)

		return Entry{}, false
	}

	return entry, true
}

// Fresh reports whether the bucket holds a live template with at least margin
// of its signed window still remaining. Probes use it to avoid re-harvesting a
// value that is not close to expiring yet.
func (s *Store) Fresh(accountID, model string, margin time.Duration) bool {
	entry, ok := s.Good(accountID, model)
	if !ok {
		return false
	}

	if margin <= 0 {
		return true
	}

	s.mu.Lock()
	ttl := s.ttl
	now := s.now()
	s.mu.Unlock()

	return now.Add(margin).Before(entry.ExpiresAt(ttl))
}

// Forget drops the template for one bucket. Callers use it when a credential
// or model binding changes and the stored value can no longer be trusted.
func (s *Store) Forget(accountID, model string) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.buckets, Bucket{
		AccountID: strings.TrimSpace(accountID),
		Model:     strings.TrimSpace(model),
	})
}

// Snapshot returns the live entries. Intended for diagnostics.
func (s *Store) Snapshot() []Entry {
	if s == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	s.purgeLocked(now)

	entries := make([]Entry, 0, len(s.buckets))
	for _, entry := range s.buckets {
		entries = append(entries, entry)
	}

	return entries
}

// purgeLocked drops expired entries. The caller must hold s.mu.
func (s *Store) purgeLocked(now time.Time) {
	for key, entry := range s.buckets {
		if !entry.live(now, s.ttl) {
			delete(s.buckets, key)
		}
	}
}
