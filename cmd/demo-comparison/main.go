package main

import (
	"fmt"
	"log"
	"os"

	"github.com/yalue/onnxruntime_go"

	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/ortinit"
)

func findFile(path string) string {
	if _, err := os.Stat(path); err == nil {
		return path
	}
	upPath := "../../" + path
	if _, err := os.Stat(upPath); err == nil {
		return upPath
	}
	return path
}

func main() {
	fmt.Println("🧪 Starting Architecture Comparison (Standard vs. Medical Expert)...")

	err := ortinit.SetupONNX()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	labels := []string{"Symptom"}
	text := "El paciente sufre de tos seca y dificultad para respirar."

	// 1. Initialize Standard Pipeline (ReLU)
	pStd, err := gliner.NewPipeline(
		findFile("encoder.onnx"),
		findFile("count_embed.onnx"),
		findFile("gliner_classifiers.safetensors"),
		findFile("tokenizer_out/tokenizer.json"),
		findFile("tests/testdata/prompt_ids.json"),
		labels,
	)
	if err != nil {
		log.Fatalf("Failed to create standard pipeline: %v", err)
	}
	defer pStd.Close()

	// 2. Initialize Medical Expert (Custom Trained)
	pMed, err := gliner.NewPipelineWithArch(
		findFile("encoder.onnx"),
		findFile("count_embed.onnx"),
		findFile("gliner_classifiers.safetensors"),
		findFile("tokenizer_out/tokenizer.json"),
		findFile("tests/testdata/prompt_ids.json"),
		labels,
		&gliner.MedicalArchitecture{WeightsPath: findFile("medical_head.safetensors")},
	)
	if err != nil {
		log.Fatalf("Failed to create Medical Expert pipeline: %v", err)
	}
	defer pMed.Close()

	fmt.Printf("\n📄 Testing Medical Text: %s\n", text)

	// Run Standard
	entsStd, _, words, _, _ := pStd.ExtractFromText(text)
	// Run Medical Expert
	entsMed, _, _, _, _ := pMed.ExtractFromText(text)

	fmt.Println("\n📊 KNOWLEDGE COMPARISON (Confidence on Symptoms):")
	fmt.Println("----------------------------------------------------------------")
	fmt.Printf("%-20s | %-18s | %-18s\n", "Word", "Standard Model", "Medical Expert")
	fmt.Println("----------------------------------------------------------------")

	// We'll iterate over the words of the text
	for i, word := range words {
		scoreStd := 0.0
		scoreMed := 0.0

		// Find if this word index was detected as an entity in either model
		for _, e := range entsStd {
			if e.Index == i {
				scoreStd = e.Score
			}
		}
		for _, e := range entsMed {
			if e.Index == i {
				scoreMed = e.Score
			}
		}

		// Only show words that have at least some minimal score in one of the models
		if scoreStd > 0.1 || scoreMed > 0.1 {
			fmt.Printf("%-20s | %-18.6f | %-18.6f\n", word, scoreStd, scoreMed)
		}
	}
	fmt.Println("----------------------------------------------------------------")
}
