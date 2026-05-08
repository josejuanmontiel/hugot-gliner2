package gliner

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

type TensorTestData struct {
	HiddenStates [][]float64 `json:"hidden_states"`
	Spans        [][2]int    `json:"spans"`
	FinalSpans   [][]float64 `json:"final_spans"`
	Scores       []float64   `json:"scores"`
}

func almostEqual(a, b, tol float64) bool {
	diff := math.Abs(a - b)
	if diff <= tol {
		return true
	}
	if math.Abs(b) > 1e-6 {
		return diff/math.Abs(b) <= 1e-4 // 0.01% relative error
	}
	return false
}

func TestGonumMathAgainstPyTorch(t *testing.T) {
	// 1. Load test data
	b, err := os.ReadFile("../../tests/testdata/tensors_test.json")
	if err != nil {
		t.Fatalf("Failed to read tensors_test.json: %v", err)
	}

	var data TensorTestData
	if err := json.Unmarshal(b, &data); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	// 2. Load Pipeline weights (We don't need ONNX for this test)
	rawTensors, err := LoadSafetensors("../../gliner_classifiers.safetensors")
	if err != nil {
		t.Fatalf("Failed to load safetensors: %v", err)
	}

	w := &ModelWeights{}
	var errExt error
	assign := func(target **mat.Dense, name string) {
		if errExt != nil {
			return
		}
		*target, errExt = getTensor(rawTensors, name)
	}

	assign(&w.SpanProjectStart0W, "span_rep.span_rep_layer.project_start.0.weight")
	assign(&w.SpanProjectStart0B, "span_rep.span_rep_layer.project_start.0.bias")
	assign(&w.SpanProjectStart3W, "span_rep.span_rep_layer.project_start.3.weight")
	assign(&w.SpanProjectStart3B, "span_rep.span_rep_layer.project_start.3.bias")

	assign(&w.SpanProjectEnd0W, "span_rep.span_rep_layer.project_end.0.weight")
	assign(&w.SpanProjectEnd0B, "span_rep.span_rep_layer.project_end.0.bias")
	assign(&w.SpanProjectEnd3W, "span_rep.span_rep_layer.project_end.3.weight")
	assign(&w.SpanProjectEnd3B, "span_rep.span_rep_layer.project_end.3.bias")

	assign(&w.SpanOutProject0W, "span_rep.span_rep_layer.out_project.0.weight")
	assign(&w.SpanOutProject0B, "span_rep.span_rep_layer.out_project.0.bias")
	assign(&w.SpanOutProject3W, "span_rep.span_rep_layer.out_project.3.weight")
	assign(&w.SpanOutProject3B, "span_rep.span_rep_layer.out_project.3.bias")

	assign(&w.Classifier0W, "classifier.0.weight")
	assign(&w.Classifier0B, "classifier.0.bias")
	assign(&w.Classifier2W, "classifier.2.weight")
	assign(&w.Classifier2B, "classifier.2.bias")

	if errExt != nil {
		t.Fatalf("Error mapping weights: %v", errExt)
	}

	pipeline := &Pipeline{
		weights: w,
	}

	// 3. Prepare input matrix
	r := len(data.HiddenStates)
	c := len(data.HiddenStates[0])
	hiddenData := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			hiddenData[i*c+j] = data.HiddenStates[i][j]
		}
	}
	hiddenStates := mat.NewDense(r, c, hiddenData)

	// 4. Run BuildSpans
	finalSpansMat, spans := pipeline.BuildSpans(hiddenStates, 8)

	// Validate number of spans
	if len(spans) != len(data.Spans) {
		t.Fatalf("Expected %d spans, got %d", len(data.Spans), len(spans))
	}

	// Validate finalSpans values
	fR, fC := finalSpansMat.Dims()
	if fR != len(data.FinalSpans) || fC != len(data.FinalSpans[0]) {
		t.Fatalf("Final spans shape mismatch: got (%d, %d), expected (%d, %d)", fR, fC, len(data.FinalSpans), len(data.FinalSpans[0]))
	}

	tol := 1e-3
	for i := 0; i < fR; i++ {
		for j := 0; j < fC; j++ {
			if !almostEqual(finalSpansMat.At(i, j), data.FinalSpans[i][j], tol) {
				t.Fatalf("finalSpans mismatch at (%d, %d): got %f, expected %f", i, j, finalSpansMat.At(i, j), data.FinalSpans[i][j])
			}
		}
	}

	t.Logf("✅ BuildSpans matches PyTorch mathematically! (Tolerance: %v)", tol)

	// 5. Run ClassifyEntities
	scores := pipeline.ClassifyEntities(finalSpansMat)

	if len(scores) != len(data.Scores) {
		t.Fatalf("Expected %d scores, got %d", len(data.Scores), len(scores))
	}

	for i := 0; i < len(scores); i++ {
		if !almostEqual(scores[i], data.Scores[i], tol) {
			t.Fatalf("Score mismatch at index %d: got %f, expected %f", i, scores[i], data.Scores[i])
		}
	}

	t.Logf("✅ ClassifyEntities matches PyTorch mathematically! (Tolerance: %v)", tol)
}
