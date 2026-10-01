package gliner

import (
	"context"
	"encoding/json"
)

// Engine is the reusable inference boundary used by the HTTP server.
type Engine interface {
	Infer(ctx context.Context, text string, schema Schema, options InferOptions) (ExtractionResult, error)
	Ready() error
}

type InferOptions struct {
	Threshold         float64
	IncludeConfidence bool
	IncludeSpans      bool
}

type EntitySpec struct {
	Name        string
	Description string
}

type ClassificationSpec struct {
	Task       string
	Labels     []string
	MultiLabel bool
	TopK       int
	Threshold  float64
}

type FieldSpec struct {
	Name        string
	DType       string
	Description string
	Choices     []string
}

type StructureSpec struct {
	Name   string
	Fields []FieldSpec
}

type RelationSpec struct {
	Name        string
	Description string
	Threshold   *float64
}

type Schema struct {
	Entities        []EntitySpec
	Classifications []ClassificationSpec
	Structures      []StructureSpec
	Relations       []RelationSpec
}

type SpanResult struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence,omitempty"`
	Start      int     `json:"start,omitempty"`
	End        int     `json:"end,omitempty"`
}

type LabelResult struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence,omitempty"`
}

type RelationMatch struct {
	Head SpanResult `json:"head"`
	Tail SpanResult `json:"tail"`
}

// ExtractionResult is normalized internally. MarshalContent applies the
// Fastino output flags and stable task buckets at the HTTP boundary.
type ExtractionResult struct {
	Entities        map[string][]SpanResult     `json:"entities"`
	Classifications map[string][]LabelResult    `json:"classifications"`
	Structures      map[string][]map[string]any `json:"structures"`
	Relations       map[string][]RelationMatch  `json:"relations"`
}

func EmptyExtractionResult(schema Schema) ExtractionResult {
	r := ExtractionResult{
		Entities:        make(map[string][]SpanResult),
		Classifications: make(map[string][]LabelResult),
		Structures:      make(map[string][]map[string]any),
		Relations:       make(map[string][]RelationMatch),
	}
	for _, entity := range schema.Entities {
		r.Entities[entity.Name] = []SpanResult{}
	}
	for _, classification := range schema.Classifications {
		r.Classifications[classification.Task] = []LabelResult{}
	}
	for _, structure := range schema.Structures {
		r.Structures[structure.Name] = []map[string]any{}
	}
	for _, relation := range schema.Relations {
		r.Relations[relation.Name] = []RelationMatch{}
	}
	return r
}

// MarshalContent produces the JSON string embedded in the assistant message.
func (r ExtractionResult) MarshalContent(includeConfidence, includeSpans bool) ([]byte, error) {
	content := map[string]any{
		"entities":        formatEntityResults(r.Entities, includeConfidence, includeSpans),
		"classifications": formatClassificationResults(r.Classifications, includeConfidence),
		"structures":      formatStructureResults(r.Structures, includeConfidence, includeSpans),
		"relations":       formatRelationResults(r.Relations, includeConfidence, includeSpans),
	}
	return json.Marshal(content)
}

func formatSpan(span SpanResult, confidence, offsets bool) any {
	if !confidence && !offsets {
		return span.Text
	}
	value := map[string]any{"text": span.Text}
	if confidence {
		value["confidence"] = span.Confidence
	}
	if offsets {
		value["start"] = span.Start
		value["end"] = span.End
	}
	return value
}

func formatEntityResults(input map[string][]SpanResult, confidence, offsets bool) any {
	out := make(map[string]any, len(input))
	for label, spans := range input {
		values := make([]any, 0, len(spans))
		for _, span := range spans {
			values = append(values, formatSpan(span, confidence, offsets))
		}
		out[label] = values
	}
	if len(out) == 0 {
		return []any{}
	}
	// Fastino's hosted chat-completions payload represents the entity task as
	// a list containing its label map, matching GLiNER2's structure instances.
	return []map[string]any{out}
}

func formatClassificationResults(input map[string][]LabelResult, confidence bool) map[string]any {
	out := make(map[string]any, len(input))
	for task, labels := range input {
		if len(labels) == 0 {
			out[task] = nil
			continue
		}
		values := make([]any, 0, len(labels))
		for _, label := range labels {
			if confidence {
				values = append(values, map[string]any{"label": label.Label, "confidence": label.Confidence})
			} else {
				values = append(values, label.Label)
			}
		}
		if len(values) == 1 {
			out[task] = values[0]
		} else {
			out[task] = values
		}
	}
	return out
}

func formatStructureResults(input map[string][]map[string]any, confidence, offsets bool) map[string]any {
	out := make(map[string]any, len(input))
	for name, records := range input {
		formatted := make([]map[string]any, 0, len(records))
		for _, record := range records {
			item := make(map[string]any, len(record))
			for field, value := range record {
				switch typed := value.(type) {
				case SpanResult:
					item[field] = formatSpan(typed, confidence, offsets)
				case []SpanResult:
					values := make([]any, 0, len(typed))
					for _, span := range typed {
						values = append(values, formatSpan(span, confidence, offsets))
					}
					item[field] = values
				default:
					item[field] = typed
				}
			}
			formatted = append(formatted, item)
		}
		out[name] = formatted
	}
	return out
}

func formatRelationResults(input map[string][]RelationMatch, confidence, offsets bool) map[string]any {
	out := make(map[string]any, len(input))
	for name, matches := range input {
		values := make([]any, 0, len(matches))
		for _, match := range matches {
			values = append(values, map[string]any{
				"head": formatSpan(match.Head, confidence, offsets),
				"tail": formatSpan(match.Tail, confidence, offsets),
			})
		}
		out[name] = values
	}
	return out
}
