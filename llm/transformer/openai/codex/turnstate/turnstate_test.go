package turnstate

import (
	"encoding/base64"
	"encoding/binary"
	"testing"
	"time"
)

// encodeToken builds a Fernet shaped value with the given decoded byte length.
// 217 bytes encode to the 292 character individual template the reference
// implementations observed; 233 bytes encode to its 312 character degraded
// counterpart (one extra AES block).
func encodeToken(issuedAt time.Time, rawLen int) string {
	raw := make([]byte, rawLen)
	raw[0] = 0x80
	if rawLen >= 9 {
		binary.BigEndian.PutUint64(raw[1:9], uint64(issuedAt.Unix()))
	}

	return base64.URLEncoding.EncodeToString(raw)
}

func TestFernetIssuedAt(t *testing.T) {
	issuedAt := time.Unix(1789600000, 0).UTC()
	value := encodeToken(issuedAt, 217)

	got, ok := FernetIssuedAt(value)
	if !ok {
		t.Fatalf("FernetIssuedAt(%q) reported the value as undecodable", value)
	}
	if !got.Equal(issuedAt) {
		t.Fatalf("FernetIssuedAt() = %s, want %s", got, issuedAt)
	}

	for _, invalid := range []string{"", "not-a-token", "AAAA", encodeToken(issuedAt, 8)} {
		if _, ok := FernetIssuedAt(invalid); ok {
			t.Fatalf("FernetIssuedAt(%q) accepted an invalid value", invalid)
		}
	}
}

func TestClassifyLengths(t *testing.T) {
	issuedAt := time.Unix(1789600000, 0).UTC()

	cases := []struct {
		name      string
		rawLen    int
		wantChars int
		want      Kind
	}{
		{name: "individual template", rawLen: 217, wantChars: 292, want: KindGood},
		{name: "team template", rawLen: 249, wantChars: 332, want: KindGood},
		{name: "individual degraded", rawLen: 233, wantChars: 312, want: KindDegraded},
		{name: "team degraded", rawLen: 267, wantChars: 356, want: KindDegraded},
		{name: "unknown size", rawLen: 100, want: KindUnknown},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value := encodeToken(issuedAt, tc.rawLen)
			if tc.wantChars != 0 && len(value) != tc.wantChars {
				t.Fatalf("encoded value length = %d, want %d", len(value), tc.wantChars)
			}
			kind, got, ok := Classify(value)
			if kind != tc.want {
				t.Fatalf("Classify() kind = %s, want %s (value length %d)", kind, tc.want, len(value))
			}
			if tc.want == KindUnknown {
				if ok {
					t.Fatal("Classify() accepted an unknown length")
				}

				return
			}
			if !ok {
				t.Fatal("Classify() rejected a recognised length")
			}
			if !got.Equal(issuedAt) {
				t.Fatalf("Classify() issuedAt = %s, want %s", got, issuedAt)
			}
		})
	}
}

func TestStoreObserveAndGood(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	value := encodeToken(now.Add(-time.Minute), 217)
	if !store.Observe("account-a", "gpt-6-astra", value) {
		t.Fatal("Observe() rejected a fresh good template")
	}

	entry, ok := store.Good("account-a", "gpt-6-astra")
	if !ok {
		t.Fatal("Good() did not return the stored template")
	}
	if entry.Value != value {
		t.Fatalf("Good() value = %q, want %q", entry.Value, value)
	}
	if entry.Kind != KindGood {
		t.Fatalf("Good() kind = %s, want %s", entry.Kind, KindGood)
	}
}

func TestStoreRejectsDegradedAndUnknown(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	if store.Observe("account-a", "gpt-6-astra", encodeToken(now, 233)) {
		t.Fatal("Observe() stored a degraded value")
	}
	if store.Observe("account-a", "gpt-6-astra", "junk") {
		t.Fatal("Observe() stored an unrecognised value")
	}
	if _, ok := store.Good("account-a", "gpt-6-astra"); ok {
		t.Fatal("Good() returned a value after only degraded observations")
	}
}

func TestStoreIsolationAndExpiry(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	value := encodeToken(now.Add(-time.Minute), 217)
	store.Observe("account-a", "gpt-6-astra", value)

	if _, ok := store.Good("account-b", "gpt-6-astra"); ok {
		t.Fatal("Good() crossed the account boundary")
	}
	if _, ok := store.Good("account-a", "gpt-5.6-sol"); ok {
		t.Fatal("Good() crossed the model boundary")
	}

	// Move the clock past the signed one hour window.
	later := now.Add(DefaultTTL + time.Minute)
	store.now = func() time.Time { return later }

	if _, ok := store.Good("account-a", "gpt-6-astra"); ok {
		t.Fatal("Good() returned an expired template")
	}
}

func TestStoreKeepsNewestIssuedValue(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	fresh := encodeToken(now.Add(-time.Minute), 217)
	stale := encodeToken(now.Add(-30*time.Minute), 217)

	store.Observe("account-a", "gpt-6-astra", fresh)
	store.Observe("account-a", "gpt-6-astra", stale)

	entry, ok := store.Good("account-a", "gpt-6-astra")
	if !ok {
		t.Fatal("Good() returned no entry")
	}
	if entry.Value != fresh {
		t.Fatal("a late, older token replaced the newer template")
	}

	// A newer token must win.
	newer := encodeToken(now.Add(-time.Second), 217)
	store.Observe("account-a", "gpt-6-astra", newer)

	entry, ok = store.Good("account-a", "gpt-6-astra")
	if !ok || entry.Value != newer {
		t.Fatal("a newer token did not replace the stored template")
	}
}

func TestStoreRejectsFutureIssuedValues(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	future := encodeToken(now.Add(10*time.Minute), 217)
	if store.Observe("account-a", "gpt-6-astra", future) {
		t.Fatal("Observe() stored a token issued far in the future")
	}

	// Within the clock skew allowance the token is accepted.
	skewed := encodeToken(now.Add(time.Minute), 217)
	if !store.Observe("account-a", "gpt-6-astra", skewed) {
		t.Fatal("Observe() rejected a token within the clock skew allowance")
	}
}

func TestStoreFresh(t *testing.T) {
	now := time.Unix(1789600000, 0).UTC()
	store := NewStore(WithClock(func() time.Time { return now }))

	// Issued 30 minutes ago, so 30 minutes of the signed hour remain.
	value := encodeToken(now.Add(-30*time.Minute), 217)
	store.Observe("account-a", "gpt-6-astra", value)

	if !store.Fresh("account-a", "gpt-6-astra", 5*time.Minute) {
		t.Fatal("Fresh() reported a template with 30 minutes left as stale")
	}
	if store.Fresh("account-a", "gpt-6-astra", 45*time.Minute) {
		t.Fatal("Fresh() reported a template with 30 minutes left as fresh for a 45 minute margin")
	}
	if store.Fresh("account-b", "gpt-6-astra", 0) {
		t.Fatal("Fresh() crossed the account boundary")
	}
}
