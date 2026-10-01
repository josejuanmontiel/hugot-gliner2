package gliner

import (
	"testing"

	"github.com/daulet/tokenizers"
)

func TestTokenizeAndPool(t *testing.T) {
	// Need to load the actual tokenizer from tests/testdata or tokenizer_out
	tk, err := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping tokenizer test, tokenizer.json not found: %v", err)
	}
	defer tk.Close()

	text := "Hello, world! This is a test."

	ids, wordIndices, words := TokenizeAndPool(tk, text)

	// "Hello", ",", "world", "!", "This", "is", "a", "test", "."
	// Expected words count: 9
	expectedWords := []string{"Hello", ",", "world", "!", "This", "is", "a", "test", "."}

	if len(words) != len(expectedWords) {
		t.Fatalf("Expected %d words, got %d", len(expectedWords), len(words))
	}

	for i, w := range expectedWords {
		if words[i] != w {
			t.Errorf("At index %d: expected %s, got %s", i, w, words[i])
		}
	}

	// Make sure word indices align properly with subword ids
	if len(wordIndices) != len(expectedWords) {
		t.Fatalf("Expected word indices len %d, got %d", len(expectedWords), len(wordIndices))
	}

	if len(ids) == 0 {
		t.Fatalf("Expected non-empty ids")
	}

	// Just checking if word indices are monotonic
	for i := 1; i < len(wordIndices); i++ {
		if wordIndices[i] <= wordIndices[i-1] {
			t.Errorf("Word indices should be strictly monotonically increasing, got %d then %d", wordIndices[i-1], wordIndices[i])
		}
	}
}

func TestTokenizeWithOffsetsUnicodeAndWhitespace(t *testing.T) {
	tk, err := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping tokenizer test, tokenizer.json not found: %v", err)
	}
	defer tk.Close()

	text := "Café  東京, India"
	_, _, words := TokenizeWithOffsets(tk, text)
	for _, word := range words {
		if got := text[word.Start:word.End]; got != word.Text {
			t.Fatalf("offsets [%d:%d] produced %q, want %q", word.Start, word.End, got, word.Text)
		}
	}
	if words[0].Start != 0 || words[0].End != len("Café") {
		t.Fatalf("unexpected UTF-8 offsets: %#v", words[0])
	}
}
