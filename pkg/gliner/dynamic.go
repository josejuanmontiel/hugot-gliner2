package gliner

import (
	"context"
	"fmt"
	stdmath "math"
	"sort"
	"strings"

	mathx "hugot-gliner2/pkg/math"

	"github.com/yalue/onnxruntime_go"
)

const (
	hiddenSize   = 768
	maxSpanLen   = 12
	maxInstances = 20
)

type dynamicTask struct {
	kind      string
	name      string
	fields    []string
	positions []int
	class     *ClassificationSpec
	structure *StructureSpec
	relation  *RelationSpec
}

func (p *Pipeline) validateDynamicTokenizer() error {
	for _, marker := range []string{"[P]", "[E]", "[C]", "[R]", "[L]", "[SEP_STRUCT]", "[SEP_TEXT]", "[DESCRIPTION]"} {
		encoded, _ := p.tk.Encode(marker, false)
		if len(encoded) != 1 {
			return fmt.Errorf("tokenizer must encode %s as one special token", marker)
		}
	}
	return nil
}

func (p *Pipeline) inferDynamic(ctx context.Context, text string, schema Schema, options InferOptions) (ExtractionResult, error) {
	result := EmptyExtractionResult(schema)
	promptIDs, tasks, err := p.buildDynamicPrompt(schema)
	if err != nil {
		return result, err
	}
	textIDs, wordIndices, words := TokenizeWithOffsets(p.tk, text)
	if len(words) == 0 {
		return result, nil
	}

	inputIDs := make([]int64, 0, len(promptIDs)+len(textIDs))
	inputIDs = append(inputIDs, promptIDs...)
	inputIDs = append(inputIDs, textIDs...)
	attention := make([]int64, len(inputIDs))
	for i := range attention {
		attention[i] = 1
	}
	hidden, err := p.RunEncoder(inputIDs, attention, 1, int64(len(inputIDs)))
	if err != nil {
		return result, fmt.Errorf("encoder inference: %w", err)
	}
	select {
	case <-ctx.Done():
		return result, ctx.Err()
	default:
	}

	wordData := make([]float32, len(words)*hiddenSize)
	for i, subword := range wordIndices {
		position := len(promptIDs) + subword
		if position < 0 || position >= len(inputIDs) {
			return result, fmt.Errorf("word/subword alignment failed at word %d", i)
		}
		copy(wordData[i*hiddenSize:(i+1)*hiddenSize], hidden[position*hiddenSize:(position+1)*hiddenSize])
	}
	wordTensor := &mathx.Tensor{Data: wordData, Shape: []int{len(words), hiddenSize}}
	spanReps, spans := p.BuildSpans(wordTensor, maxSpanLen)

	for _, task := range tasks {
		if len(task.positions) != len(task.fields)+1 {
			return result, fmt.Errorf("schema embedding layout mismatch for %s", task.name)
		}
		embeddings := make([]float32, len(task.positions)*hiddenSize)
		for i, position := range task.positions {
			copy(embeddings[i*hiddenSize:(i+1)*hiddenSize], hidden[position*hiddenSize:(position+1)*hiddenSize])
		}
		if task.kind == "classifications" {
			result.Classifications[task.name] = p.decodeClassification(embeddings[hiddenSize:], *task.class)
			continue
		}
		count := p.predictCount(embeddings[:hiddenSize])
		if count <= 0 {
			continue
		}
		projection, err := p.runCountEmbed(embeddings[hiddenSize:], len(task.fields))
		if err != nil {
			return result, fmt.Errorf("count embedding for %s: %w", task.name, err)
		}
		scores := scoreSpans(spanReps, projection, count, len(task.fields))
		switch task.kind {
		case "entities":
			for fieldIndex, name := range task.fields {
				result.Entities[name] = decodeSpanList(scores[0][fieldIndex], spans, words, text, options.Threshold, false)
			}
		case "structures":
			result.Structures[task.name] = decodeStructures(scores, spans, words, text, *task.structure, options.Threshold)
		case "relations":
			threshold := options.Threshold
			if task.relation.Threshold != nil {
				threshold = *task.relation.Threshold
			}
			result.Relations[task.name] = decodeRelations(scores, spans, words, text, threshold)
		}
	}
	return result, nil
}

func (p *Pipeline) buildDynamicPrompt(schema Schema) ([]int64, []dynamicTask, error) {
	tasks := make([]dynamicTask, 0, len(schema.Entities)+len(schema.Classifications)+len(schema.Structures)+len(schema.Relations))
	tokenBlocks := make([][]string, 0, cap(tasks))
	appendTask := func(task dynamicTask, marker, promptDescription string, descriptions []string) {
		promptText := task.name
		if promptDescription != "" {
			promptText += ": " + promptDescription
		}
		for i, field := range task.fields {
			if i < len(descriptions) && descriptions[i] != "" {
				promptText += " [DESCRIPTION] " + field + ": " + descriptions[i]
			}
		}
		tokens := []string{"(", "[P]", promptText, "("}
		for _, field := range task.fields {
			tokens = append(tokens, marker, field)
		}
		tokens = append(tokens, ")", ")")
		tasks = append(tasks, task)
		tokenBlocks = append(tokenBlocks, tokens)
	}

	for i := range schema.Structures {
		structure := &schema.Structures[i]
		fields, descriptions := make([]string, len(structure.Fields)), make([]string, len(structure.Fields))
		for j, field := range structure.Fields {
			fields[j], descriptions[j] = field.Name, field.Description
		}
		appendTask(dynamicTask{kind: "structures", name: structure.Name, fields: fields, structure: structure}, "[C]", "", descriptions)
	}
	if len(schema.Entities) > 0 {
		fields, descriptions := make([]string, len(schema.Entities)), make([]string, len(schema.Entities))
		for i, entity := range schema.Entities {
			fields[i], descriptions[i] = entity.Name, entity.Description
		}
		appendTask(dynamicTask{kind: "entities", name: "entities", fields: fields}, "[E]", "", descriptions)
	}
	for i := range schema.Relations {
		relation := &schema.Relations[i]
		appendTask(dynamicTask{kind: "relations", name: relation.Name, fields: []string{"head", "tail"}, relation: relation}, "[R]", relation.Description, nil)
	}
	for i := range schema.Classifications {
		classification := &schema.Classifications[i]
		appendTask(dynamicTask{kind: "classifications", name: classification.Task, fields: classification.Labels, class: classification}, "[L]", "", nil)
	}
	var ids []int64
	for blockIndex, block := range tokenBlocks {
		if blockIndex > 0 {
			ids = appendEncoded(p, ids, "[SEP_STRUCT]")
		}
		for tokenIndex, token := range block {
			position := len(ids)
			ids = appendEncoded(p, ids, token)
			if tokenIndex == 1 || (tokenIndex >= 4 && tokenIndex < len(block)-2 && tokenIndex%2 == 0) {
				tasks[blockIndex].positions = append(tasks[blockIndex].positions, position)
			}
		}
	}
	ids = appendEncoded(p, ids, "[SEP_TEXT]")
	return ids, tasks, nil
}

func appendEncoded(p *Pipeline, destination []int64, token string) []int64 {
	encoded, _ := p.tk.Encode(token, false)
	for _, id := range encoded {
		destination = append(destination, int64(id))
	}
	return destination
}

func (p *Pipeline) predictCount(parentEmbedding []float32) int {
	logits := p.Heads.CountPredictor.Forward(&mathx.Tensor{Data: append([]float32(nil), parentEmbedding...), Shape: []int{1, hiddenSize}})
	best := 0
	for i := 1; i < len(logits.Data) && i <= maxInstances; i++ {
		if logits.Data[i] > logits.Data[best] {
			best = i
		}
	}
	return best
}

func (p *Pipeline) decodeClassification(fieldEmbeddings []float32, spec ClassificationSpec) []LabelResult {
	count := len(spec.Labels)
	logits := p.Heads.Classification.Forward(&mathx.Tensor{Data: append([]float32(nil), fieldEmbeddings...), Shape: []int{count, hiddenSize}}).Data
	results := make([]LabelResult, 0, count)
	if spec.MultiLabel {
		for i, label := range spec.Labels {
			probability := sigmoid(float64(logits[i]))
			if probability >= spec.Threshold {
				results = append(results, LabelResult{Label: label, Confidence: probability})
			}
		}
	} else {
		probabilities := softmax(logits)
		indices := make([]int, count)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool { return probabilities[indices[i]] > probabilities[indices[j]] })
		limit := spec.TopK
		if limit < 1 {
			limit = 1
		}
		for _, index := range indices[:limit] {
			results = append(results, LabelResult{Label: spec.Labels[index], Confidence: probabilities[index]})
		}
	}
	return results
}

func (p *Pipeline) runCountEmbed(fieldEmbeddings []float32, numFields int) ([]float32, error) {
	input, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{int64(numFields), hiddenSize}, fieldEmbeddings)
	if err != nil {
		return nil, err
	}
	defer input.Destroy()
	output, err := onnxruntime_go.NewEmptyTensor[float32](onnxruntime_go.Shape{maxInstances, int64(numFields), hiddenSize})
	if err != nil {
		return nil, err
	}
	defer output.Destroy()
	if err := p.countSession.Run([]onnxruntime_go.Value{input}, []onnxruntime_go.Value{output}); err != nil {
		return nil, err
	}
	return append([]float32(nil), output.GetData()...), nil
}

func scoreSpans(spans *mathx.Tensor, projection []float32, count, fields int) [][][]float64 {
	spanCount, _ := spans.Dims()
	result := make([][][]float64, count)
	for instance := 0; instance < count; instance++ {
		result[instance] = make([][]float64, fields)
		for field := 0; field < fields; field++ {
			result[instance][field] = make([]float64, spanCount)
			projected := projection[(instance*fields+field)*hiddenSize : (instance*fields+field+1)*hiddenSize]
			for span := 0; span < spanCount; span++ {
				vector := spans.Data[span*hiddenSize : (span+1)*hiddenSize]
				result[instance][field][span] = sigmoid(float64(mathx.Dot(vector, projected)))
			}
		}
	}
	return result
}

func decodeSpanList(scores []float64, spans [][2]int, words []WordToken, original string, threshold float64, single bool) []SpanResult {
	indices := NMS(spans, scores, threshold)
	if single && len(indices) > 1 {
		indices = indices[:1]
	}
	result := make([]SpanResult, 0, len(indices))
	for _, index := range indices {
		span := spans[index]
		if span[0] < 0 || span[1] >= len(words) {
			continue
		}
		start, end := words[span[0]].Start, words[span[1]].End
		result = append(result, SpanResult{Text: original[start:end], Confidence: scores[index], Start: start, End: end})
	}
	return result
}

func decodeStructures(scores [][][]float64, spans [][2]int, words []WordToken, original string, spec StructureSpec, threshold float64) []map[string]any {
	result := make([]map[string]any, 0, len(scores))
	seen := make(map[string]struct{})
	for instance := range scores {
		record := make(map[string]any, len(spec.Fields))
		nonempty := false
		var key strings.Builder
		for fieldIndex, field := range spec.Fields {
			matches := decodeSpanList(scores[instance][fieldIndex], spans, words, original, threshold, field.DType != "list")
			if field.DType == "list" {
				record[field.Name] = matches
			} else if len(matches) > 0 {
				record[field.Name] = matches[0]
			} else {
				record[field.Name] = nil
			}
			for _, match := range matches {
				key.WriteString(fmt.Sprintf("%d:%d;", match.Start, match.End))
				nonempty = true
			}
		}
		if nonempty {
			if _, exists := seen[key.String()]; !exists {
				seen[key.String()] = struct{}{}
				result = append(result, record)
			}
		}
	}
	return result
}

func decodeRelations(scores [][][]float64, spans [][2]int, words []WordToken, original string, threshold float64) []RelationMatch {
	result := make([]RelationMatch, 0, len(scores))
	seen := make(map[string]struct{})
	for instance := range scores {
		if len(scores[instance]) < 2 {
			continue
		}
		head := decodeSpanList(scores[instance][0], spans, words, original, threshold, true)
		tail := decodeSpanList(scores[instance][1], spans, words, original, threshold, true)
		if len(head) == 0 || len(tail) == 0 {
			continue
		}
		key := fmt.Sprintf("%d:%d-%d:%d", head[0].Start, head[0].End, tail[0].Start, tail[0].End)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, RelationMatch{Head: head[0], Tail: tail[0]})
	}
	return result
}

func sigmoid(value float64) float64 { return 1 / (1 + stdmath.Exp(-value)) }
func argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
func softmax(values []float32) []float64 {
	result := make([]float64, len(values))
	max := values[argmax(values)]
	var total float64
	for i, value := range values {
		result[i] = stdmath.Exp(float64(value - max))
		total += result[i]
	}
	for i := range result {
		result[i] /= total
	}
	return result
}
