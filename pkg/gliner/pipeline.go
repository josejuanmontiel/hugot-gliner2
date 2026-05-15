package gliner

import (
	"encoding/json"
	"fmt"
	"os"

	"hugot-gliner2/pkg/math"
	"hugot-gliner2/pkg/types"

	"github.com/daulet/tokenizers"
	"github.com/yalue/onnxruntime_go"
)

// Pipeline represents the GLiNER inference pipeline.
type Pipeline struct {
	encoderSession *onnxruntime_go.DynamicAdvancedSession
	countSession   *onnxruntime_go.DynamicAdvancedSession

	// Modular Classification Heads
	Heads *Heads

	labels       []string
	tk           *tokenizers.Tokenizer
	promptIDs    []int64
	promptLabels []string
}

// NewPipeline creates a pipeline using the standard GLiNER2 architecture.
func NewPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath string, labels []string) (*Pipeline, error) {
	return NewPipelineWithArch(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath, labels, &GLiNER2Architecture{})
}

// NewPipelineWithArch creates a pipeline using a specific head architecture strategy.
func NewPipelineWithArch(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath string, labels []string, arch HeadArchitecture) (*Pipeline, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	encoderSession, err := onnxruntime_go.NewDynamicAdvancedSession(encoderPath,
		[]string{"input_ids", "attention_mask"}, []string{"last_hidden_state"}, nil)
	if err != nil {
		tk.Close()
		return nil, fmt.Errorf("failed to create encoder session: %w", err)
	}

	countSession, err := onnxruntime_go.NewDynamicAdvancedSession(countPath,
		[]string{"pc_emb"}, []string{"struct_proj"}, nil)
	if err != nil {
		tk.Close()
		encoderSession.Destroy()
		return nil, fmt.Errorf("failed to create count session: %w", err)
	}

	promptData, err := LoadPromptData(promptPath)
	if err != nil {
		tk.Close()
		encoderSession.Destroy()
		countSession.Destroy()
		return nil, fmt.Errorf("failed to load prompt data: %w", err)
	}

	rawTensors, err := LoadSafetensors(safetensorsPath)
	if err != nil {
		tk.Close()
		encoderSession.Destroy()
		countSession.Destroy()
		return nil, fmt.Errorf("failed to load safetensors: %w", err)
	}

	heads, err := arch.Assemble(rawTensors)
	if err != nil {
		tk.Close()
		encoderSession.Destroy()
		countSession.Destroy()
		return nil, fmt.Errorf("failed to assemble heads: %w", err)
	}

	p := &Pipeline{
		encoderSession: encoderSession,
		countSession:   countSession,
		Heads:          heads,
		labels:         promptData.Labels,
		tk:             tk,
		promptIDs:      promptData.PromptIDs,
		promptLabels:   promptData.Labels,
	}

	return p, nil
}

// GetTokenizer returns the internal tokenizer.
func (p *Pipeline) GetTokenizer() *tokenizers.Tokenizer {
	return p.tk
}

// GetPromptIDs returns the internal prompt IDs.
func (p *Pipeline) GetPromptIDs() []int64 {
	return p.promptIDs
}

// Close cleans up the ONNX sessions and tokenizer.
func (p *Pipeline) Close() {
	if p.tk != nil {
		p.tk.Close()
	}
	if p.encoderSession != nil {
		p.encoderSession.Destroy()
	}
	if p.countSession != nil {
		p.countSession.Destroy()
	}
}

// RunEncoder executes the Transformer backbone.
func (p *Pipeline) RunEncoder(inputIDs, attentionMask []int64, batchSize, seqLen int64) ([]float32, error) {
	inputTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.Shape{batchSize, seqLen}, inputIDs)
	maskTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.Shape{batchSize, seqLen}, attentionMask)
	defer inputTensor.Destroy()
	defer maskTensor.Destroy()

	outputTensor, _ := onnxruntime_go.NewEmptyTensor[float32](onnxruntime_go.Shape{batchSize, seqLen, 768})
	defer outputTensor.Destroy()

	err := p.encoderSession.Run([]onnxruntime_go.Value{inputTensor, maskTensor}, []onnxruntime_go.Value{outputTensor})
	if err != nil {
		return nil, err
	}

	return outputTensor.GetData(), nil
}

// ExtractEntities handles word pooling and classification heads.
func (p *Pipeline) ExtractEntities(wordMat *math.Tensor, maxSpanLen int) (*math.Tensor, [][2]int, []float64) {
	spanReps, spansInfo := p.BuildSpans(wordMat, maxSpanLen)
	scores := p.ClassifyEntities(spanReps)
	return spanReps, spansInfo, scores
}

// ExtractRelations computes relationship scores using the count_embed backbone.
func (p *Pipeline) ExtractRelations(spanReps *math.Tensor, spansInfo [][2]int, validIndices []int, pcEmb []float32, numFields int64) ([]types.RelationResult, error) {
	hiddenSize := 768
	pcTensor, _ := onnxruntime_go.NewTensor(onnxruntime_go.Shape{numFields, int64(hiddenSize)}, pcEmb)
	defer pcTensor.Destroy()

	outputTensor, _ := onnxruntime_go.NewEmptyTensor[float32](onnxruntime_go.Shape{numFields, 2, int64(hiddenSize)})
	defer outputTensor.Destroy()

	err := p.countSession.Run([]onnxruntime_go.Value{pcTensor}, []onnxruntime_go.Value{outputTensor})
	if err != nil {
		return nil, err
	}
	
	structProj := outputTensor.GetData()
	maxCount := int(numFields)
	relations := ExtractRelations(spanReps, spansInfo, validIndices, structProj, maxCount, 2, hiddenSize)
	return relations, nil
}

// ExtractFromText provides an E2E high-level API.
func (p *Pipeline) ExtractFromText(text string) ([]types.SpanMatch, []types.RelationResult, []string, [][2]int, error) {
	idsArr, wordIndices, words := TokenizeAndPool(p.tk, text)
	
	fullInputIDs := append(p.promptIDs, idsArr...)
	fullAttentionMask := make([]int64, len(fullInputIDs))
	for i := range fullAttentionMask {
		fullAttentionMask[i] = 1
	}

	batchSize := int64(1)
	seqLen := int64(len(fullInputIDs))
	
	hiddenStatesData, err := p.RunEncoder(fullInputIDs, fullAttentionMask, batchSize, seqLen)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	promptLen := len(p.promptIDs)
	hiddenSize := 768
	
	// Pool
	wordF32 := make([]float32, len(words)*hiddenSize)
	for i, tokenIdx := range wordIndices {
		for d := 0; d < hiddenSize; d++ {
			wordF32[i*hiddenSize+d] = hiddenStatesData[(promptLen+tokenIdx)*hiddenSize+d]
		}
	}
	wordMat := &math.Tensor{Data: wordF32, Shape: []int{len(words), hiddenSize}}

	// Entities (Standard threshold 0.5)
	spanReps, spansInfo, scores := p.ExtractEntities(wordMat, 12)
	validIndices := NMS(spansInfo, scores, 0.5)

	var entities []types.SpanMatch
	for _, idx := range validIndices {
		entities = append(entities, types.SpanMatch{
			Index: idx,
			Score: scores[idx],
			Label: "Entity",
		})
	}

	// Relations
	numRelations := int64(len(p.promptLabels))
	if numRelations > 0 {
		pcEmb := make([]float32, numRelations*int64(hiddenSize))
		// Map back to the [R] tokens in the prompt (usually one per label)
		// For demo simplicity, we take the first few tokens. 
		// In production, this would be mapped via the prompt_ids logic.
		copy(pcEmb, hiddenStatesData[0:numRelations*int64(hiddenSize)])
		
		relations, err := p.ExtractRelations(spanReps, spansInfo, validIndices, pcEmb, numRelations)
		if err == nil {
			for i := range relations {
				if i < len(p.promptLabels) {
					relations[i].Label = p.promptLabels[i]
				}
			}
			return entities, relations, words, spansInfo, nil
		}
	}

	return entities, nil, words, spansInfo, nil
}

// PromptData represents pre-calculated prompt IDs and labels.
type PromptData struct {
	PromptIDs []int64  `json:"prompt_ids"`
	Labels    []string `json:"labels"`
}

// LoadPromptData loads prompt config from JSON.
func LoadPromptData(path string) (*PromptData, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pd PromptData
	err = json.Unmarshal(data, &pd)
	return &pd, err
}
