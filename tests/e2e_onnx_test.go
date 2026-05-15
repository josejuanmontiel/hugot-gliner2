package gliner_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/yalue/onnxruntime_go"

	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/math"
	"hugot-gliner2/pkg/ortinit"
)

// TestE2EPipelineExample is an example test showing how to assemble
// the ONNX Runtime step and the modular math step in Go using the public API.
func TestE2EPipelineExample(t *testing.T) {
	// 1. Initialize ONNX environment
	err := ortinit.SetupONNX()
	if err != nil {
		t.Skipf("Skipping ONNX test because C library is not configured: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 2. Instantiate our Pipeline
	pipeline, err := gliner.NewPipeline(
		"../encoder.onnx",
		"../count_embed.onnx",
		"../gliner_classifiers.safetensors",
		"../tokenizer_out/tokenizer.json",
		"../tests/testdata/prompt_ids.json",
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to create pipeline: %v", err)
	}
	defer pipeline.Close()

	// 3. Load Simulation Data (hugot_sim.json)
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
		t.Fatalf("Could not read hugot_sim.json: %v", err)
	}
	var sim HugotSim
	if err := json.Unmarshal(b, &sim); err != nil {
		t.Fatalf("JSON parse error: %v", err)
	}

	batchSize := int64(1)
	seqLen := int64(len(sim.InputIds))

	// 4. ONNX ENCODER EXECUTION
	hiddenStatesData, err := pipeline.RunEncoder(sim.InputIds, sim.AttentionMask, batchSize, seqLen)
	if err != nil {
		t.Fatalf("Encoder inference failed: %v", err)
	}

	// 5. TOKEN POOLING TO WORDS ("first" pooling)
	numWords := len(sim.TextWordIndices)
	wordF32 := make([]float32, numWords*768)
	for i, tokenIdx := range sim.TextWordIndices {
		for d := 0; d < 768; d++ {
			wordF32[i*768+d] = hiddenStatesData[tokenIdx*768+d]
		}
	}
	wordMat := &math.Tensor{Data: wordF32, Shape: []int{numWords, 768}}

	// 6. ENTITY EXTRACTION
	spanReps, spansInfo, scores := pipeline.ExtractEntities(wordMat, 12)
	validIndices := gliner.NMS(spansInfo, scores, 0.999)

	fmt.Printf("\n=======================================\n")
	fmt.Printf("🏆 SUCCESSFUL GO-GLINER E2E INFERENCE! 🏆\n")
	fmt.Printf("=======================================\n")
	fmt.Printf("High-confidence candidate entities found:\n")

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

	// 7. RELATION EXTRACTION
	fmt.Printf("\nEvaluating Relations...\n")

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

		// validIndices can be nil to extract relations for all possible spans
		relations, err := pipeline.ExtractRelations(spanReps, spansInfo, nil, pcEmbFlat, numFields)
		if err != nil {
			t.Fatalf("Relation inference failed: %v", err)
		}

		for _, rel := range relations {
			headTxt := getText(rel.Head.Index)
			tailTxt := getText(rel.Tail.Index)

			fmt.Printf(" 🤝 [%s] ---> %s ---> [%s] (Confidence: H=%.4f, T=%.4f)\n",
				headTxt, label, tailTxt, rel.Head.Score, rel.Tail.Score)
		}
	}
}
