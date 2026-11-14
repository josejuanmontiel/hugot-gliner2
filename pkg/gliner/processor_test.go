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

func TestBuildRelationPromptIDsAndFormat(t *testing.T) {
	tk, err := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping tokenizer test, tokenizer.json not found: %v", err)
	}
	defer tk.Close()

	labels := []string{"trabaja en", "fundó", "adquirió"}
	promptIDs, rPos := BuildRelationPromptIDs(tk, labels)
	if len(rPos) != len(labels) {
		t.Fatalf("Expected %d relation positions, got %d", len(labels), len(rPos))
	}
	if promptIDs[len(promptIDs)-1] != tokenSepText {
		t.Fatalf("Expected trailing SEP_TEXT token %d, got %d", tokenSepText, promptIDs[len(promptIDs)-1])
	}
	found := FindRelationRPositions(promptIDs)
	if len(found) != len(labels) {
		t.Fatalf("FindRelationRPositions mismatch: got %d want %d", len(found), len(labels))
	}
	for i := range rPos {
		if rPos[i] != found[i] {
			t.Errorf("Position mismatch at %d: %v vs %v", i, rPos[i], found[i])
		}
	}

	words := []string{"Elena", "Rodríguez", ",", "CEOs", "(", "Madrid", ")"}
	if got := FormatSpanText(words, [2]int{0, 1}); got != "Elena Rodríguez" {
		t.Errorf("Expected 'Elena Rodríguez', got %q", got)
	}
	if got := FormatSpanText(words, [2]int{4, 6}); got != "(Madrid)" {
		t.Errorf("Expected '(Madrid)', got %q", got)
	}
}
