package gliner

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/daulet/tokenizers"

	mathx "hugot-gliner2/pkg/math"
)

type fixedModule struct{ output []float32 }

func (m fixedModule) Forward(*mathx.Tensor) *mathx.Tensor {
	return &mathx.Tensor{Data: append([]float32(nil), m.output...), Shape: []int{len(m.output), 1}}
}

func TestBuildDynamicPromptCombinedSchema(t *testing.T) {
	tk, err := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping tokenizer test, tokenizer.json not found: %v", err)
	}
	defer tk.Close()
	pipeline := &Pipeline{tk: tk}
	if err := pipeline.validateDynamicTokenizer(); err != nil {
		t.Fatal(err)
	}

	schema := Schema{
		Entities:        []EntitySpec{{Name: "person"}, {Name: "company", Description: "business name"}},
		Structures:      []StructureSpec{{Name: "product", Fields: []FieldSpec{{Name: "name", DType: "str"}, {Name: "features", DType: "list"}}}},
		Relations:       []RelationSpec{{Name: "works_for"}},
		Classifications: []ClassificationSpec{{Task: "sentiment", Labels: []string{"positive", "negative"}, TopK: 1}},
	}
	ids, tasks, err := pipeline.buildDynamicPrompt(schema)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 || len(tasks) != 4 {
		t.Fatalf("unexpected prompt: ids=%d tasks=%#v", len(ids), tasks)
	}
	for _, task := range tasks {
		if len(task.positions) != len(task.fields)+1 {
			t.Fatalf("task %s has invalid marker positions: %#v", task.name, task.positions)
		}
		for _, position := range task.positions {
			if position < 0 || position >= len(ids) {
				t.Fatalf("task %s marker out of range: %d", task.name, position)
			}
		}
	}
}

func TestDynamicPromptMatchesUpstreamGolden(t *testing.T) {
	tk, err := tokenizers.FromFile("../../tokenizer_out/tokenizer.json")
	if err != nil {
		t.Skipf("Skipping tokenizer test, tokenizer.json not found: %v", err)
	}
	defer tk.Close()

	data, err := os.ReadFile("../../tests/testdata/prompt_ids.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden PromptData
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	schema := Schema{Relations: make([]RelationSpec, len(golden.Labels))}
	for i, label := range golden.Labels {
		schema.Relations[i] = RelationSpec{Name: label}
	}
	ids, _, err := (&Pipeline{tk: tk}).buildDynamicPrompt(schema)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, golden.PromptIDs) {
		t.Fatalf("dynamic prompt diverged from upstream golden\n got: %v\nwant: %v", ids, golden.PromptIDs)
	}
}

func TestClassificationDecoding(t *testing.T) {
	pipeline := &Pipeline{Heads: &Heads{Classification: fixedModule{output: []float32{0, 1}}}}
	single := pipeline.decodeClassification(make([]float32, 2*hiddenSize), ClassificationSpec{
		Labels: []string{"negative", "positive"}, TopK: 1,
	})
	if len(single) != 1 || single[0].Label != "positive" {
		t.Fatalf("unexpected single-label result: %#v", single)
	}
	want := 1 / (1 + math.Exp(-1))
	if math.Abs(single[0].Confidence-want) > 1e-9 {
		t.Fatalf("softmax confidence = %v, want %v", single[0].Confidence, want)
	}

	pipeline.Heads.Classification = fixedModule{output: []float32{-2, 2}}
	multi := pipeline.decodeClassification(make([]float32, 2*hiddenSize), ClassificationSpec{
		Labels: []string{"a", "b"}, MultiLabel: true, Threshold: 0.9,
	})
	if len(multi) != 0 {
		t.Fatalf("multi-label threshold should permit an empty result: %#v", multi)
	}
}

func TestCountPredictionAllowsTwentyInstances(t *testing.T) {
	logits := make([]float32, maxInstances+1)
	logits[maxInstances] = 10
	pipeline := &Pipeline{Heads: &Heads{CountPredictor: fixedModule{output: logits}}}
	if got := pipeline.predictCount(make([]float32, hiddenSize)); got != maxInstances {
		t.Fatalf("count = %d, want %d", got, maxInstances)
	}
}

func TestDecodeSpanListUsesOriginalByteOffsets(t *testing.T) {
	original := "Go 😊 Zürich"
	words := []WordToken{{Text: "Go", Start: 0, End: 2}, {Text: "😊", Start: 3, End: 7}, {Text: "Zürich", Start: 8, End: 15}}
	got := decodeSpanList([]float64{0.95}, [][2]int{{1, 2}}, words, original, 0.5, true)
	if len(got) != 1 || got[0].Text != "😊 Zürich" || got[0].Start != 3 || got[0].End != 15 {
		t.Fatalf("unexpected decoded span: %#v", got)
	}
}
