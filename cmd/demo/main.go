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

	// 1. Setup ONNX Environment
	err := ortinit.SetupONNX()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Instantiate our Pipeline (Expects assets in project root)
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

	// 4. Extract entities and relations
	entities, relations, words, _, err := pipeline.ExtractFromText(text)
	if err != nil {
		log.Fatalf("Extraction failed: %v", err)
	}

	// 5. Display results
	fmt.Println("🏷️ Extracted Entities:")
	for _, entity := range entities {
		if entity.Index < len(words) {
			fmt.Printf("   🔹 %s (score: %.4f)\n", words[entity.Index], entity.Score)
		}
	}

	fmt.Println("\n🤝 Extracted Relations:")
	for _, rel := range relations {
		headWord := "unknown"
		tailWord := "unknown"
		if rel.Head.Index < len(words) {
			headWord = words[rel.Head.Index]
		}
		if rel.Tail.Index < len(words) {
			tailWord = words[rel.Tail.Index]
		}
		fmt.Printf("   🔗 [%s] --- %s ---> [%s]\n", headWord, rel.Label, tailWord)
	}

	fmt.Println("\n✅ Demo Finished.")
}
