package main

import (
	"fmt"
	"log"

	"github.com/yalue/onnxruntime_go"

	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/ortinit"
)

func main() {
	fmt.Println("🚀 Initializing GLiNER2 Pipeline Demo (E2E Text-to-Relations)...")

	// 1. Initialize ONNX runtime environment robustly
	err := ortinit.SetupONNX()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Instantiate our Pipeline with Tokenizer and Prompt IDs
	pipeline, err := gliner.NewPipeline(
		"encoder.onnx",
		"count_embed.onnx",
		"gliner_classifiers.safetensors",
		"tokenizer_out/tokenizer.json",
		"tests/testdata/prompt_ids.json",
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipeline.Close()

	// 3. Define raw text
	text := "Last Monday, Elena Rodriguez, Chief Operating Officer of TechNova Solutions, announced in Madrid the acquisition of the Finnish startup Nordic AI. According to the agreement valued at 45 million euros, the founder of Nordic AI, Lukas Virtanen, will join the executive committee of TechNova. This operation was supervised by Banco Santander, which acted as the main financial advisor, ensuring that the technological integration begins next June in its Helsinki offices."
	
	fmt.Printf("\n📄 Processing original text:\n%s\n\n", text)
	fmt.Println("🧠 Running E2E pipeline (Tokenization -> ONNX -> Modular Math)...")

	// 4. Run E2E Inference
	entities, relations, words, spansInfo, err := pipeline.ExtractFromText(text)
	if err != nil {
		log.Fatalf("Extraction failed: %v", err)
	}

	getText := func(spanIdx int) string {
		span := spansInfo[spanIdx]
		startWord, endWord := span[0], span[1]
		txt := ""
		for w := startWord; w <= endWord; w++ {
			if w > startWord {
				txt += " "
			}
			txt += words[w]
		}
		return txt
	}

	fmt.Printf("\n✨ Candidate Entities:\n")
	for _, ent := range entities {
		fmt.Printf("   🔹 %s (score: %.4f)\n", getText(ent.Index), ent.Score)
	}

	fmt.Printf("\n🤝 Extracted Relations:\n")
	for _, rel := range relations {
		fmt.Printf("   [%s] ---> %s ---> [%s] (Confidence: H=%.4f, T=%.4f)\n",
			getText(rel.Head.Index), rel.Label, getText(rel.Tail.Index), rel.Head.Score, rel.Tail.Score)
	}
	
	fmt.Println("\n✅ Demo Finished.")
}
