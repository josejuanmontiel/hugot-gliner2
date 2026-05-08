package gliner_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/mat"

	"hugot-gliner2/pkg/gliner"
)

// TestE2EPipelineExample es un test de ejemplo para mostrar cómo se ensambla
// el paso de ONNX Runtime y el paso matemático de Gonum en Go usando la API pública.
// Nota: Este test asume que la librería compartida de onnxruntime está instalada en el sistema.
func TestE2EPipelineExample(t *testing.T) {
	// 1. Iniciar el entorno de ONNX
	onnxruntime_go.SetSharedLibraryPath("/home/jose/openvino/openvino/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.so.1.25.1")
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		t.Skipf("Saltando test de ONNX porque no está configurada la librería C: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Instanciar nuestro Pipeline
	pipeline, err := gliner.NewPipeline(
		"../encoder.onnx",
		"../count_embed.onnx",
		"../gliner_classifiers.safetensors",
		nil,
	)
	if err != nil {
		t.Fatalf("No se pudo crear el pipeline: %v", err)
	}
	defer pipeline.Close()

	// 3. Cargar la simulación del Tokenizador (hugot_sim.json)
	type HugotSim struct {
		InputIds        []int64       `json:"input_ids"`
		AttentionMask   []int64       `json:"attention_mask"`
		TextWordIndices []int         `json:"text_word_indices"`
		TextTokens      []string      `json:"text_tokens"`
		PcEmbs          [][][]float32 `json:"pc_embs"`
		RelationLabels  []string      `json:"relation_labels"`
	}

	b, err := os.ReadFile("testdata/hugot_sim.json")
	if err != nil {
		t.Fatalf("No se pudo leer hugot_sim.json: %v", err)
	}
	var sim HugotSim
	if err := json.Unmarshal(b, &sim); err != nil {
		t.Fatalf("JSON parse error: %v", err)
	}

	batchSize := int64(1)
	seqLen := int64(len(sim.InputIds))

	// 4. EJECUCIÓN ONNX ENCODER
	hiddenStatesData, err := pipeline.RunEncoder(sim.InputIds, sim.AttentionMask, batchSize, seqLen)
	if err != nil {
		t.Fatalf("Fallo en la inferencia del encoder: %v", err)
	}

	// 5. POOLING DE TOKENS A PALABRAS ("first" pooling)
	numWords := len(sim.TextWordIndices)
	wordF64 := make([]float64, numWords*768)
	for i, tokenIdx := range sim.TextWordIndices {
		for d := 0; d < 768; d++ {
			val := hiddenStatesData[tokenIdx*768+d]
			wordF64[i*768+d] = float64(val)
		}
	}
	wordMat := mat.NewDense(numWords, 768, wordF64)

	// 6. EXTRACCIÓN DE ENTIDADES
	spanReps, spansInfo, scores := pipeline.ExtractEntities(wordMat, 12)
	validIndices := gliner.NMS(spansInfo, scores, 0.999)

	fmt.Printf("\n=======================================\n")
	fmt.Printf("🏆 ¡INFERENCIA E2E GO-GLINER EXITOSA! 🏆\n")
	fmt.Printf("=======================================\n")
	fmt.Printf("Entidades candidatas de muy alta confianza encontradas:\n")

	for _, idx := range validIndices {
		span := spansInfo[idx]
		score := scores[idx]

		startWord := span[0]
		endWord := span[1]

		if startWord < 20 {
			continue
		}

		extractedText := ""
		for w := startWord; w <= endWord; w++ {
			if w > startWord && sim.TextTokens[w] != "," && sim.TextTokens[w] != "." {
				extractedText += " "
			}
			extractedText += sim.TextTokens[w]
		}

		if len(extractedText) > 2 {
			fmt.Printf(" 🔹 %s (score: %.4f)\n", extractedText, score)
		}
	}

	// 7. EXTRACCIÓN DE RELACIONES
	fmt.Printf("\nEvaluando Relaciones...\n")

	getText := func(spanIdx int) string {
		span := spansInfo[spanIdx]
		startWord, endWord := span[0], span[1]
		txt := ""
		for w := startWord; w <= endWord; w++ {
			if w > startWord && sim.TextTokens[w] != "," && sim.TextTokens[w] != "." {
				txt += " "
			}
			txt += sim.TextTokens[w]
		}
		return txt
	}

	for i, pcEmb := range sim.PcEmbs {
		label := sim.RelationLabels[i]

		pcEmbFlat := make([]float32, 0, len(pcEmb)*768)
		for _, row := range pcEmb {
			pcEmbFlat = append(pcEmbFlat, row...)
		}
		numFields := int64(len(pcEmb))

		relations, err := pipeline.ExtractRelations(spanReps, spansInfo, pcEmbFlat, numFields)
		if err != nil {
			t.Fatalf("Fallo en inferencia de relations: %v", err)
		}

		for _, rel := range relations {
			headTxt := getText(rel.Head.Index)
			tailTxt := getText(rel.Tail.Index)

			fmt.Printf(" 🤝 [%s] ---> %s ---> [%s] (Confianza: H=%.4f, T=%.4f)\n",
				headTxt, label, tailTxt, rel.Head.Score, rel.Tail.Score)
		}
	}
}
