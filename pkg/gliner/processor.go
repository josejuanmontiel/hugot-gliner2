package gliner

import (
	"regexp"
	"strings"

	"github.com/daulet/tokenizers"
)

const (
	tokenParenOpen  int64 = 287
	tokenParenClose int64 = 1263
	tokenHead       int64 = 761
	tokenTail       int64 = 6214
	tokenUNK        int64 = 3
	tokenSepStruct  int64 = 128001
	tokenSepText    int64 = 128002
	tokenPromptP    int64 = 128003
	tokenRoleR      int64 = 128006
)

var wordSplitRegex = regexp.MustCompile(`[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`)

// TokenizeAndPool processes raw text into subword IDs, word start indices, and the extracted words.
// It encodes word-by-word so wordIndices always align 1:1 with the first subword of each word.
func TokenizeAndPool(tk *tokenizers.Tokenizer, text string) ([]int64, []int, []string) {
	words := wordSplitRegex.FindAllString(text, -1)
	if len(words) == 0 {
		return nil, nil, nil
	}

	ids := make([]int64, 0, len(words)*2)
	wordIndices := make([]int, len(words))

	for i, w := range words {
		wordIndices[i] = len(ids)
		subIDs, _ := tk.Encode(strings.ToLower(w), false)
		if len(subIDs) == 0 {
			subIDs, _ = tk.Encode(w, false)
		}
		if len(subIDs) == 0 {
			ids = append(ids, tokenUNK)
			continue
		}
		for _, id := range subIDs {
			ids = append(ids, int64(id))
		}
	}

	return ids, wordIndices, words
}

// BuildRelationPromptIDs dynamically constructs GLiNER2 Cross-Encoder relation prompt token IDs
// for a given list of relation labels and returns the [headRPos, tailRPos] token positions for each label.
func BuildRelationPromptIDs(tk *tokenizers.Tokenizer, labels []string) ([]int64, [][2]int) {
	if len(labels) == 0 {
		return nil, nil
	}

	promptIDs := make([]int64, 0, len(labels)*14)
	rPositions := make([][2]int, 0, len(labels))

	for i, lbl := range labels {
		promptIDs = append(promptIDs, tokenParenOpen, tokenPromptP)
		lblIDs, _ := tk.Encode(strings.ToLower(strings.TrimSpace(lbl)), false)
		for _, id := range lblIDs {
			promptIDs = append(promptIDs, int64(id))
		}
		promptIDs = append(promptIDs, tokenParenOpen)
		headRPos := len(promptIDs)
		promptIDs = append(promptIDs, tokenRoleR, tokenHead)
		tailRPos := len(promptIDs)
		promptIDs = append(promptIDs, tokenRoleR, tokenTail, tokenParenClose, tokenParenClose)

		rPositions = append(rPositions, [2]int{headRPos, tailRPos})

		if i < len(labels)-1 {
			promptIDs = append(promptIDs, tokenSepStruct)
		} else {
			promptIDs = append(promptIDs, tokenSepText)
		}
	}

	return promptIDs, rPositions
}

// FindRelationRPositions locates pairs of [R] (128006) token positions inside a precomputed prompt IDs slice.
func FindRelationRPositions(promptIDs []int64) [][2]int {
	var rIndices []int
	for idx, id := range promptIDs {
		if id == tokenRoleR {
			rIndices = append(rIndices, idx)
		}
	}

	positions := make([][2]int, 0, len(rIndices)/2)
	for i := 0; i+1 < len(rIndices); i += 2 {
		positions = append(positions, [2]int{rIndices[i], rIndices[i+1]})
	}
	return positions
}

// FormatSpanText reconstructs a clean multi-word string from words[span[0]:span[1]+1].
func FormatSpanText(words []string, span [2]int) string {
	start, end := span[0], span[1]
	if start < 0 || end >= len(words) || start > end {
		return ""
	}
	var b strings.Builder
	for w := start; w <= end; w++ {
		tok := words[w]
		if w > start && !noSpaceBefore(tok) && !noSpaceAfter(words[w-1]) {
			b.WriteByte(' ')
		}
		b.WriteString(tok)
	}
	return strings.TrimSpace(b.String())
}

func noSpaceBefore(tok string) bool {
	switch tok {
	case ",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "”", "»":
		return true
	}
	return false
}

func noSpaceAfter(tok string) bool {
	switch tok {
	case "(", "[", "{", "¡", "¿", "“", "«":
		return true
	}
	return false
}
