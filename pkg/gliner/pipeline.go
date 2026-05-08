package gliner

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/daulet/tokenizers"
	"github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/mat"
)

// ModelWeights holds the Gonum matrices for the classification heads.
type ModelWeights struct {
	SpanProjectStart0W *mat.Dense
	SpanProjectStart0B *mat.Dense
	SpanProjectStart3W *mat.Dense
	SpanProjectStart3B *mat.Dense

	SpanProjectEnd0W *mat.Dense
	SpanProjectEnd0B *mat.Dense
	SpanProjectEnd3W *mat.Dense
	SpanProjectEnd3B *mat.Dense

	SpanOutProject0W *mat.Dense
	SpanOutProject0B *mat.Dense
	SpanOutProject3W *mat.Dense
	SpanOutProject3B *mat.Dense

	Classifier0W *mat.Dense
	Classifier0B *mat.Dense
	Classifier2W *mat.Dense
	Classifier2B *mat.Dense
}

// Pipeline represents the GLiNER inference pipeline.
type Pipeline struct {
	encoderSession *onnxruntime_go.DynamicAdvancedSession
	countSession   *onnxruntime_go.DynamicAdvancedSession
	weights        *ModelWeights
	labels         []string // The entity/relation labels
	tk             *tokenizers.Tokenizer
	promptIDs      []int64
	promptLabels   []string
}

// getTensor is a helper to safely extract a tensor by name.
func getTensor(tensors map[string]*mat.Dense, name string) (*mat.Dense, error) {
	t, ok := tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor: %s", name)
	}
	return t, nil
}

// NewPipeline initializes a new GLiNER pipeline.
func NewPipeline(encoderPath, countEmbedPath, safetensorsPath, tokenizerPath, promptIDsPath string, labels []string) (*Pipeline, error) {
	// Load tokenizer
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Load prompt ids
	b, err := os.ReadFile(promptIDsPath)
	if err != nil {
		tk.Close()
		return nil, fmt.Errorf("failed to load prompt ids: %w", err)
	}
	var promptData struct {
		PromptIDs []int64  `json:"prompt_ids"`
		Labels    []string `json:"labels"`
	}
	if err := json.Unmarshal(b, &promptData); err != nil {
		tk.Close()
		return nil, fmt.Errorf("failed to parse prompt ids: %w", err)
	}

	// Configure dynamic session options
	sessionOptions, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("error creating session options: %w", err)
	}
	defer sessionOptions.Destroy()

	encoderSession, err := onnxruntime_go.NewDynamicAdvancedSession(
		encoderPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating encoder session: %w", err)
	}

	countSession, err := onnxruntime_go.NewDynamicAdvancedSession(
		countEmbedPath,
		[]string{"pc_emb"},
		[]string{"struct_proj"},
		sessionOptions,
	)
	if err != nil {
		encoderSession.Destroy()
		return nil, fmt.Errorf("error creating count_embed session: %w", err)
	}

	// 2. Read safetensors and populate ModelWeights.
	rawTensors, err := LoadSafetensors(safetensorsPath)
	if err != nil {
		encoderSession.Destroy()
		countSession.Destroy()
		return nil, fmt.Errorf("failed to load safetensors: %w", err)
	}

	w := &ModelWeights{}
	var errExt error

	// Helper to extract and assign
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
		encoderSession.Destroy()
		countSession.Destroy()
		tk.Close()
		return nil, fmt.Errorf("error mapping weights: %w", errExt)
	}

	return &Pipeline{
		encoderSession: encoderSession,
		countSession:   countSession,
		weights:        w,
		labels:         promptData.Labels,
		tk:             tk,
		promptIDs:      promptData.PromptIDs,
		promptLabels:   promptData.Labels,
	}, nil
}

// Close cleans up the ONNX session.
func (p *Pipeline) Close() error {
	var err1, err2 error
	if p.encoderSession != nil {
		err1 = p.encoderSession.Destroy()
	}
	if p.countSession != nil {
		err2 = p.countSession.Destroy()
	}
	if p.tk != nil {
		p.tk.Close()
	}
	if err1 != nil {
		return err1
	}
	return err2
}

// RunEncoder executes the ONNX encoder and returns the hidden states.
// inputIds and attentionMask should be flattened slices representing batchSize x seqLen.
func (p *Pipeline) RunEncoder(inputIds, attentionMask []int64, batchSize, seqLen int64) ([]float32, error) {
	inputIdsTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(batchSize, seqLen), inputIds)
	if err != nil {
		return nil, fmt.Errorf("error creating input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	attentionTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.NewShape(batchSize, seqLen), attentionMask)
	if err != nil {
		return nil, fmt.Errorf("error creating attention_mask tensor: %w", err)
	}
	defer attentionTensor.Destroy()

	outShape := onnxruntime_go.NewShape(batchSize, seqLen, 768)
	hiddenStatesData := make([]float32, batchSize*seqLen*768)
	hiddenTensor, err := onnxruntime_go.NewTensor(outShape, hiddenStatesData)
	if err != nil {
		return nil, fmt.Errorf("error creating output tensor: %w", err)
	}
	defer hiddenTensor.Destroy()

	err = p.encoderSession.Run(
		[]onnxruntime_go.Value{inputIdsTensor, attentionTensor},
		[]onnxruntime_go.Value{hiddenTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("error running encoder session: %w", err)
	}

	return hiddenStatesData, nil
}

// ExtractEntities processes the pooled word representations to find entity candidates.
// wordMat is the [numWords x 768] matrix obtained after pooling tokens to words.
func (p *Pipeline) ExtractEntities(wordMat *mat.Dense, maxSpanLen int) (*mat.Dense, [][2]int, []float64) {
	spanReps, spansInfo := p.BuildSpans(wordMat, maxSpanLen)
	scores := p.ClassifyEntities(spanReps)
	return spanReps, spansInfo, scores
}

// ExtractRelations queries the count_embed model and extracts relations.
// pcEmb is a flattened slice of float32 representing the relation schema embeddings [numFields x 768].
func (p *Pipeline) ExtractRelations(spanReps *mat.Dense, spansInfo [][2]int, validIndices []int, pcEmb []float32, numFields int64) ([]RelationResult, error) {
	pcEmbTensor, err := onnxruntime_go.NewTensor(
		onnxruntime_go.NewShape(numFields, 768),
		pcEmb,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating pc_emb tensor: %w", err)
	}
	defer pcEmbTensor.Destroy()

	maxCount := int64(20)
	structProjShape := onnxruntime_go.NewShape(maxCount, numFields, 768)
	structProjData := make([]float32, maxCount*numFields*768)
	structProjTensor, err := onnxruntime_go.NewTensor(structProjShape, structProjData)
	if err != nil {
		return nil, fmt.Errorf("error creating struct_proj tensor: %w", err)
	}
	defer structProjTensor.Destroy()

	err = p.countSession.Run(
		[]onnxruntime_go.Value{pcEmbTensor},
		[]onnxruntime_go.Value{structProjTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("error running count_embed session: %w", err)
	}

	relations := ExtractRelations(spanReps, spansInfo, validIndices, structProjData, int(maxCount), int(numFields), 768)
	return relations, nil
}

// ExtractFromText processes a raw text string, tokenizes it, runs the E2E pipeline, and returns extracted entities and relations.
func (p *Pipeline) ExtractFromText(text string) ([]SpanMatch, []RelationResult, []string, [][2]int, error) {
	// 1. Tokenize and pool
	textIDs, wordIndices, words := TokenizeAndPool(p.tk, text)

	// 2. Combine prompt + text
	inputIds := make([]int64, 0, len(p.promptIDs)+len(textIDs))
	inputIds = append(inputIds, p.promptIDs...)
	inputIds = append(inputIds, textIDs...)

	seqLen := int64(len(inputIds))
	attentionMask := make([]int64, seqLen)
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	// 3. Run Encoder
	hiddenStatesData, err := p.RunEncoder(inputIds, attentionMask, 1, seqLen)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("encoder failed: %w", err)
	}

	// 4. Extract pc_embs from prompt
	// The prompt has `[R]` tokens. We need to find their indices.
	// 128006 is the ID for `[R]` in gliner2-base-v1 tokenizer
	var rIndices []int
	for i, id := range p.promptIDs {
		if id == 128006 {
			rIndices = append(rIndices, i)
		}
	}

	numLabels := len(p.promptLabels)
	numFields := int64(2) // head, tail
	
	pcEmbsFlat := make([]float32, 0, numLabels*int(numFields)*768)
	
	// Collect pc_embs for all labels
	// Each label has 2 [R] tokens
	for i := 0; i < numLabels; i++ {
		r1Idx := rIndices[i*2]
		r2Idx := rIndices[i*2+1]
		
		for d := 0; d < 768; d++ {
			pcEmbsFlat = append(pcEmbsFlat, hiddenStatesData[r1Idx*768+d])
		}
		for d := 0; d < 768; d++ {
			pcEmbsFlat = append(pcEmbsFlat, hiddenStatesData[r2Idx*768+d])
		}
	}

	// 5. Pool text to words
	numWords := len(wordIndices)
	wordF64 := make([]float64, numWords*768)
	promptLen := len(p.promptIDs)
	
	for i, tokenIdx := range wordIndices {
		// Offset by prompt length
		actualIdx := promptLen + tokenIdx
		for d := 0; d < 768; d++ {
			val := hiddenStatesData[actualIdx*768+d]
			wordF64[i*768+d] = float64(val)
		}
	}
	wordMat := mat.NewDense(numWords, 768, wordF64)

	// 6. Extract Entities
	spanReps, spansInfo, scores := p.ExtractEntities(wordMat, 12)
	validIndices := NMS(spansInfo, scores, 0.5) // sensible threshold for flat NER
	
	var entities []SpanMatch
	for _, idx := range validIndices {
		entities = append(entities, SpanMatch{Index: idx, Score: scores[idx]})
	}

	// 7. Extract Relations
	var allRelations []RelationResult
	
	for i := 0; i < numLabels; i++ {
		start := i * int(numFields) * 768
		end := start + int(numFields)*768
		labelPcEmb := pcEmbsFlat[start:end]
		
		// Note: we pass ALL spans (not just validIndices) so we don't miss relations 
		// whose entities were missed by the flat NER classifier.
		relations, err := p.ExtractRelations(spanReps, spansInfo, nil, labelPcEmb, numFields)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("extract relations failed for label %d: %w", i, err)
		}
		
		// Map the relation label into the RelationResult
		for iRel := range relations {
			relations[iRel].Label = p.promptLabels[i]
		}
		
		// Filter overlapping relations for THIS label using greedy NMS
		var filtered []RelationResult
		for _, rel := range relations {
			overlap := false
			for _, f := range filtered {
				// We say two relations overlap if BOTH head and tail overlap
				headOverlap := rel.Head.Index == f.Head.Index || 
					(spansInfo[rel.Head.Index][0] <= spansInfo[f.Head.Index][1] && spansInfo[rel.Head.Index][1] >= spansInfo[f.Head.Index][0])
				tailOverlap := rel.Tail.Index == f.Tail.Index ||
					(spansInfo[rel.Tail.Index][0] <= spansInfo[f.Tail.Index][1] && spansInfo[rel.Tail.Index][1] >= spansInfo[f.Tail.Index][0])
				
				if headOverlap && tailOverlap {
					overlap = true
					break
				}
			}
			if !overlap {
				filtered = append(filtered, rel)
			}
		}

		allRelations = append(allRelations, filtered...)
	}

	return entities, allRelations, words, spansInfo, nil
}
