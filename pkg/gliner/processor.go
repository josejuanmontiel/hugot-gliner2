package gliner

import (
	"regexp"

	"github.com/daulet/tokenizers"
)

type WordToken struct {
	Text       string
	Start, End int
}

var wordPattern = regexp.MustCompile(`[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`)

// TokenizeWithOffsets preserves half-open byte offsets into the original UTF-8
// string while retaining the first-subword pooling convention used by GLiNER2.
func TokenizeWithOffsets(tk *tokenizers.Tokenizer, text string) ([]int64, []int, []WordToken) {
	indices := wordPattern.FindAllStringIndex(text, -1)
	words := make([]WordToken, 0, len(indices))
	for _, index := range indices {
		words = append(words, WordToken{Text: text[index[0]:index[1]], Start: index[0], End: index[1]})
	}
	var ids []int64
	wordIndices := make([]int, len(words))
	for wordIndex, token := range words {
		wordIndices[wordIndex] = len(ids)
		encoded, _ := tk.Encode(token.Text, false)
		for _, id := range encoded {
			ids = append(ids, int64(id))
		}
	}
	return ids, wordIndices, words
}

// TokenizeAndPool processes raw text into subword IDs, word start indices, and the extracted words.
func TokenizeAndPool(tk *tokenizers.Tokenizer, text string) ([]int64, []int, []string) {
	ids, wordIndices, tokens := TokenizeWithOffsets(tk, text)
	words := make([]string, len(tokens))
	for i, token := range tokens {
		words[i] = token.Text
	}
	return ids, wordIndices, words
}
