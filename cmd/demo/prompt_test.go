package main

import (
	"fmt"
	"testing"
	"github.com/daulet/tokenizers"
)

func BuildPrompt(tk *tokenizers.Tokenizer, labels []string) []int64 {
	var ids []int64
	for i, label := range labels {
		ids = append(ids, 287, 128003)
		
		// Encode label
		labelIDsUint, _ := tk.Encode(label, false)
		for _, v := range labelIDsUint {
			ids = append(ids, int64(v))
		}
		
		// Append relation structure
		ids = append(ids, 287, 128006, 761, 128006, 6214, 1263, 1263)
		
		if i == len(labels)-1 {
			ids = append(ids, 128002) // SEP_TEXT
		} else {
			ids = append(ids, 128001) // SEP_STRUCT
		}
	}
	return ids
}

func TestPromptBuilder(t *testing.T) {
	tk, _ := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	defer tk.Close()
	
	labels := []string{"trabaja en", "ubicado en", "fundó", "adquirió"}
	
	ids := BuildPrompt(tk, labels)
	fmt.Printf("Dynamic IDs: %v\n", ids)
}
