package gliner

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestNormalizeSchemaCombined(t *testing.T) {
	raw := json.RawMessage(`{
		"entities":[{"name":"organization","description":"business name"},"person"],
		"classifications":{"sentiment":{"labels":["positive","negative"],"top_k":1}},
		"structures":{"product":["name::str::product name","features::list::features"]},
		"relations":{"works_for":{"description":"employment","threshold":0.6}}
	}`)
	schema, deprecated, err := NormalizeSchema(raw)
	if err != nil {
		t.Fatalf("NormalizeSchema() error = %v", err)
	}
	if deprecated {
		t.Fatal("canonical object schema should not be deprecated")
	}
	if len(schema.Entities) != 2 || len(schema.Classifications) != 1 || len(schema.Structures) != 1 || len(schema.Relations) != 1 {
		t.Fatalf("unexpected normalized schema: %#v", schema)
	}
	if schema.Structures[0].Fields[1].DType != "list" {
		t.Fatalf("field dtype not preserved: %#v", schema.Structures[0])
	}
	if schema.Relations[0].Threshold == nil || *schema.Relations[0].Threshold != 0.6 {
		t.Fatalf("relation threshold not preserved")
	}
}

func TestNormalizeSchemaRejectsExcessiveCardinality(t *testing.T) {
	labels := make([]string, MaxSchemaFields+1)
	for i := range labels {
		labels[i] = fmt.Sprintf("label-%d", i)
	}
	raw := json.RawMessage(fmt.Sprintf(`{"entities":["%s"]}`, strings.Join(labels, `","`)))
	if _, _, err := NormalizeSchema(raw); err == nil {
		t.Fatal("expected schema cardinality error")
	}
}

func TestNormalizeSchemaLegacyFlatEntities(t *testing.T) {
	schema, deprecated, err := NormalizeSchema(json.RawMessage(`["person","company"]`))
	if err != nil {
		t.Fatal(err)
	}
	if !deprecated {
		t.Fatal("flat schema should be marked deprecated")
	}
	if len(schema.Entities) != 2 {
		t.Fatalf("got %d entities", len(schema.Entities))
	}
}

func TestNormalizeSchemaRejectsInvalidInputs(t *testing.T) {
	tests := []string{
		`{}`,
		`{"entities":["person","person"]}`,
		`{"classifications":[{"task":"sentiment","labels":["positive"]}]}`,
		`{"structures":{"item":["value::integer"]}}`,
		`{"relations":{"works_for":{"threshold":2}}}`,
	}
	for _, input := range tests {
		t.Run(input, func(t *testing.T) {
			if _, _, err := NormalizeSchema(json.RawMessage(input)); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestMarshalContentFlags(t *testing.T) {
	result := ExtractionResult{
		Entities:        map[string][]SpanResult{"person": {{Text: "Ada", Confidence: .9, Start: 0, End: 3}}},
		Classifications: map[string][]LabelResult{"sentiment": {{Label: "positive", Confidence: .8}}},
		Structures:      map[string][]map[string]any{},
		Relations:       map[string][]RelationMatch{"works_for": {{Head: SpanResult{Text: "Ada", Confidence: .9, Start: 0, End: 3}, Tail: SpanResult{Text: "Acme", Confidence: .7, Start: 7, End: 11}}}},
	}
	raw, err := result.MarshalContent(false, false)
	if err != nil {
		t.Fatal(err)
	}
	var content map[string]any
	if err := json.Unmarshal(raw, &content); err != nil {
		t.Fatal(err)
	}
	entities := content["entities"].([]any)[0].(map[string]any)["person"].([]any)
	if entities[0] != "Ada" {
		t.Fatalf("unexpected entity payload: %#v", entities)
	}
	relations := content["relations"].(map[string]any)["works_for"].([]any)
	if relations[0].(map[string]any)["head"] != "Ada" {
		t.Fatalf("unexpected relation payload: %#v", relations)
	}
}
