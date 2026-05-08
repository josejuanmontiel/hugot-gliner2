package main

import (
	"fmt"
	"strings"
	"testing"
	"github.com/daulet/tokenizers"
	"regexp"
)

func TestTokenizer(t *testing.T) {
	tk, _ := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	defer tk.Close()
	
	text := "El pasado lunes, Elena Rodríguez, Directora de Operaciones."
	ids, tokens := tk.Encode(text, false)
	
	re := regexp.MustCompile(`[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`)
	words := re.FindAllString(text, -1)
	
	fmt.Printf("Words: %v\n", words)
	fmt.Printf("Subwords: %v\n", tokens)
	fmt.Printf("IDs: %v\n", ids)
	
	wordIndices := make([]int, len(words))
	wordIdx := 0
	subwordIdx := 0
	
	for wordIdx < len(words) {
		word := strings.ToLower(words[wordIdx])
		accum := ""
		firstIdx := subwordIdx
		
		for subwordIdx < len(tokens) {
			subword := strings.ReplaceAll(tokens[subwordIdx], " ", "") 
			subword = strings.ToLower(subword)
			accum += subword
			subwordIdx++
			if accum == word {
				break
			}
			if len(accum) > len(word) {
				fmt.Printf("Mismatch: word=%s, accum=%s\n", word, accum)
			    break
			}
		}
		wordIndices[wordIdx] = firstIdx
		wordIdx++
	}
	
	fmt.Printf("Word Indices: %v\n", wordIndices)
}
