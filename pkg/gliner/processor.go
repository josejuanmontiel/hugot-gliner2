package gliner

import (
	"regexp"
	"strings"

	"github.com/daulet/tokenizers"
)

// TokenizeAndPool processes raw text into subword IDs, word start indices, and the extracted words.
func TokenizeAndPool(tk *tokenizers.Tokenizer, text string) ([]int64, []int, []string) {
	// 1. Get subwords and IDs
	// Encode(text, addSpecialTokens) - we don't add special tokens because we append to prompt
	idsUint, subwordStrs := tk.Encode(text, false)
	
	ids := make([]int64, len(idsUint))
	for i, v := range idsUint {
		ids[i] = int64(v)
	}

	// 2. Split words
	re := regexp.MustCompile(`[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`)
	words := re.FindAllString(text, -1)

	// 3. Align words with subwords to find TextWordIndices
	wordIndices := make([]int, len(words))
	wordIdx := 0
	subwordIdx := 0

	for wordIdx < len(words) {
		word := strings.ToLower(words[wordIdx])
		accum := ""
		firstIdx := subwordIdx

		for subwordIdx < len(subwordStrs) {
			sub := subwordStrs[subwordIdx]
			sub = strings.ReplaceAll(sub, "▁", "")
			sub = strings.ToLower(sub)
			accum += sub
			subwordIdx++
			
			if accum == word {
				break
			}
			if len(accum) > len(word) {
				// Mismatch heuristic: break and let the next word try to align
				break
			}
		}
		wordIndices[wordIdx] = firstIdx
		wordIdx++
	}

	return ids, wordIndices, words
}
