package main

import (
	"fmt"
	"log"

	"github.com/yalue/onnxruntime_go"

	"hugot-gliner2/pkg/gliner"
)

func main() {
	fmt.Println("🚀 Initializing GLiNER2 Pipeline Demo (E2E Text-to-Relations)...")

	// 1. Initialize ONNX runtime environment
	// IMPORTANT: Update this path to match your local onnxruntime installation!
	onnxruntime_go.SetSharedLibraryPath("/home/jose/openvino/openvino/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.so.1.25.1")
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v\nMake sure the shared library path is correct.", err)
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
	text := "El pasado lunes, Elena Rodríguez, Directora de Operaciones de TechNova Solutions, anunció en la sede de Madrid la adquisición de la startup finlandesa Nordic AI. Según el acuerdo valorado en 45 millones de euros, el fundador de Nordic AI, Lukas Virtanen, se unirá al comité ejecutivo de TechNova. Esta operación fue supervisada por el Banco Santander, que actuó como asesor financiero principal, asegurando que la integración tecnológica comience el próximo mes de junio en sus oficinas de Helsinki."
	
	fmt.Printf("\n📄 Procesando texto original:\n%s\n\n", text)
	fmt.Println("🧠 Ejecutando pipeline E2E (Tokenización -> ONNX -> Gonum)...")

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

	fmt.Printf("\n✨ Entidades Candidatas:\n")
	for _, ent := range entities {
		fmt.Printf("   🔹 %s (score: %.4f)\n", getText(ent.Index), ent.Score)
	}

	fmt.Printf("\n🤝 Relaciones Extraídas:\n")
	for _, rel := range relations {
		fmt.Printf("   [%s] ---> %s ---> [%s] (Confianza: H=%.4f, T=%.4f)\n",
			getText(rel.Head.Index), rel.Label, getText(rel.Tail.Index), rel.Head.Score, rel.Tail.Score)
	}
	
	fmt.Println("\n✅ Demo Finalizada.")
}
