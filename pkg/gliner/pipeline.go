package gliner

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode"

	"hugot-gliner2/pkg/math"
	"hugot-gliner2/pkg/types"

	"github.com/daulet/tokenizers"
	"github.com/yalue/onnxruntime_go"
)

const (
	hiddenDim      = 768
	maxEncoderSeq  = 2048
	defaultRelConf = 0.50
)

// Pipeline represents the GLiNER inference pipeline.
type Pipeline struct {
	mu             sync.Mutex
	encoderSession *onnxruntime_go.DynamicAdvancedSession
	countSession   *onnxruntime_go.DynamicAdvancedSession

	// Modular Classification Heads
	Heads *Heads

	labels        []string
	tk            *tokenizers.Tokenizer
	promptIDs     []int64
	promptLabels  []string
	relRPositions [][2]int
}

// NewPipeline creates a pipeline using the standard GLiNER2 architecture.
func NewPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath string, labels []string, options *onnxruntime_go.SessionOptions) (*Pipeline, error) {
	return NewPipelineWithArch(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath, labels, &GLiNER2Architecture{}, options)
}

// NewPipelineWithArch creates a pipeline using a specific head architecture strategy.
func NewPipelineWithArch(encoderPath, countPath, safetensorsPath, tokenizerPath, promptPath string, labels []string, arch HeadArchitecture, options *onnxruntime_go.SessionOptions) (*Pipeline, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	encoderSession, err := onnxruntime_go.NewDynamicAdvancedSession(encoderPath,
		[]string{"input_ids", "attention_mask"}, []string{"last_hidden_state"}, options)
	if err != nil {
		tk.Close()
		return nil, fmt.Errorf("failed to create encoder session: %w", err)
	}

	countSession, err := onnxruntime_go.NewDynamicAdvancedSession(countPath,
		[]string{"pc_emb"}, []string{"struct_proj"}, options)
	if err != nil {
		tk.Close()
		encoderSession.Destroy()
		return nil, fmt.Errorf("failed to create count session: %w", err)
	}

	var promptIDs []int64
	var promptLabels []string
	var relRPositions [][2]int

	if len(labels) > 0 {
		promptLabels = append([]string(nil), labels...)
		promptIDs, relRPositions = BuildRelationPromptIDs(tk, promptLabels)
	} else {
		promptData, err := LoadPromptData(promptPath)
		if err != nil {
			tk.Close()
			encoderSession.Destroy()
			countSession.Destroy()
			return nil, fmt.Errorf("failed to load prompt data: %w", err)
		}
		promptIDs = promptData.PromptIDs
		promptLabels = promptData.Labels
		relRPositions = FindRelationRPositions(promptIDs)
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
		labels:         promptLabels,
		tk:             tk,
		promptIDs:      promptIDs,
		promptLabels:   promptLabels,
		relRPositions:  relRPositions,
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

// GetLabels returns the active relation labels configured in this pipeline.
func (p *Pipeline) GetLabels() []string {
	return append([]string(nil), p.promptLabels...)
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
	inputTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{batchSize, seqLen}, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	maskTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{batchSize, seqLen}, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	outputTensor, err := onnxruntime_go.NewEmptyTensor[float32](onnxruntime_go.Shape{batchSize, seqLen, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = p.encoderSession.Run([]onnxruntime_go.Value{inputTensor, maskTensor}, []onnxruntime_go.Value{outputTensor})
	if err != nil {
		return nil, err
	}

	src := outputTensor.GetData()
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst, nil
}

// ExtractEntities handles word pooling and classification heads.
func (p *Pipeline) ExtractEntities(wordMat *math.Tensor, maxSpanLen int) (*math.Tensor, [][2]int, []float64) {
	spanReps, spansInfo := p.BuildSpans(wordMat, maxSpanLen)
	scores := p.ClassifyEntities(spanReps)
	return spanReps, spansInfo, scores
}

// ExtractRelations computes relationship scores using the count_embed backbone.
func (p *Pipeline) ExtractRelations(spanReps *math.Tensor, spansInfo [][2]int, validIndices []int, pcEmb []float32, numFields int64) ([]types.RelationResult, error) {
	pcTensor, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{numFields, int64(hiddenDim)}, pcEmb)
	if err != nil {
		return nil, fmt.Errorf("pc tensor: %w", err)
	}
	defer pcTensor.Destroy()

	// maxCount represents the fixed maximum number of relations exported in the ONNX model.
	const maxCount = 20
	outputTensor, err := onnxruntime_go.NewEmptyTensor[float32](onnxruntime_go.Shape{maxCount, numFields, int64(hiddenDim)})
	if err != nil {
		return nil, fmt.Errorf("count output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = p.countSession.Run([]onnxruntime_go.Value{pcTensor}, []onnxruntime_go.Value{outputTensor})
	if err != nil {
		return nil, err
	}

	structProj := outputTensor.GetData()
	relations := ExtractRelations(spanReps, spansInfo, validIndices, structProj, maxCount, int(numFields), hiddenDim)
	return relations, nil
}

// ExtractFromText provides an E2E high-level API for entity and relation extraction.
func (p *Pipeline) ExtractFromText(text string) ([]types.SpanMatch, []types.RelationResult, []string, [][2]int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	idsArr, wordIndices, words := TokenizeAndPool(p.tk, text)
	if len(words) == 0 || len(idsArr) == 0 {
		return nil, nil, nil, nil, nil
	}

	// Enforce DeBERTa-v3 max context length by truncating extra trailing words.
	maxTextTokens := maxEncoderSeq - len(p.promptIDs)
	if maxTextTokens < 16 {
		return nil, nil, nil, nil, fmt.Errorf("prompt too long (%d tokens) for max sequence %d", len(p.promptIDs), maxEncoderSeq)
	}
	if len(idsArr) > maxTextTokens {
		cutoffWords := 0
		for i, tokIdx := range wordIndices {
			if tokIdx >= maxTextTokens {
				break
			}
			cutoffWords = i + 1
		}
		if cutoffWords == 0 {
			cutoffWords = 1
		}
		words = words[:cutoffWords]
		wordIndices = wordIndices[:cutoffWords]
		idsArr = idsArr[:maxTextTokens]
	}

	fullInputIDs := make([]int64, 0, len(p.promptIDs)+len(idsArr))
	fullInputIDs = append(fullInputIDs, p.promptIDs...)
	fullInputIDs = append(fullInputIDs, idsArr...)

	fullAttentionMask := make([]int64, len(fullInputIDs))
	for i := range fullAttentionMask {
		fullAttentionMask[i] = 1
	}

	hiddenStatesData, err := p.RunEncoder(fullInputIDs, fullAttentionMask, 1, int64(len(fullInputIDs)))
	if err != nil {
		return nil, nil, nil, nil, err
	}

	promptLen := len(p.promptIDs)

	// Pool word representations using first subword index
	wordF32 := make([]float32, len(words)*hiddenDim)
	for i, tokenIdx := range wordIndices {
		if tokenIdx >= len(idsArr) {
			tokenIdx = len(idsArr) - 1
		}
		srcStart := (promptLen + tokenIdx) * hiddenDim
		copy(wordF32[i*hiddenDim:(i+1)*hiddenDim], hiddenStatesData[srcStart:srcStart+hiddenDim])
	}
	wordMat := &math.Tensor{Data: wordF32, Shape: []int{len(words), hiddenDim}}

	// Entities (NMS threshold 0.5)
	spanReps, spansInfo, scores := p.ExtractEntities(wordMat, 12)
	validIndices := NMS(spansInfo, scores, 0.5)

	var entities []types.SpanMatch
	for _, idx := range validIndices {
		txt := FormatSpanText(words, spansInfo[idx])
		if !isMeaningfulSpan(txt) {
			continue
		}
		entities = append(entities, types.SpanMatch{
			Index: idx,
			Score: scores[idx],
			Label: "Entity",
			Text:  txt,
		})
	}

	// Relations: extract the [R] head and [R] tail contextual embeddings for each configured relation label.
	var allRelations []types.RelationResult
	seenRel := make(map[string]bool)

	for r, label := range p.promptLabels {
		if r >= len(p.relRPositions) {
			break
		}
		headRPos := p.relRPositions[r][0]
		tailRPos := p.relRPositions[r][1]
		if headRPos < 0 || tailRPos < 0 || tailRPos >= promptLen {
			continue
		}

		pcEmb := make([]float32, 2*hiddenDim)
		copy(pcEmb[0:hiddenDim], hiddenStatesData[headRPos*hiddenDim:(headRPos+1)*hiddenDim])
		copy(pcEmb[hiddenDim:2*hiddenDim], hiddenStatesData[tailRPos*hiddenDim:(tailRPos+1)*hiddenDim])

		rels, err := p.ExtractRelations(spanReps, spansInfo, nil, pcEmb, 2)
		if err != nil {
			continue
		}

		for _, rel := range rels {
			if rel.Head.Score < defaultRelConf || rel.Tail.Score < defaultRelConf {
				continue
			}
			if rel.Head.Index < 0 || rel.Head.Index >= len(spansInfo) || rel.Tail.Index < 0 || rel.Tail.Index >= len(spansInfo) {
				continue
			}
			if rel.Head.Index == rel.Tail.Index {
				continue
			}

			headTxt := FormatSpanText(words, spansInfo[rel.Head.Index])
			tailTxt := FormatSpanText(words, spansInfo[rel.Tail.Index])
			if !isMeaningfulSpan(headTxt) || !isMeaningfulSpan(tailTxt) {
				continue
			}
			if strings.EqualFold(headTxt, tailTxt) {
				continue
			}

			dedupKey := strings.ToLower(headTxt) + "|" + strings.ToLower(label) + "|" + strings.ToLower(tailTxt)
			if seenRel[dedupKey] {
				continue
			}
			seenRel[dedupKey] = true

			rel.Label = label
			rel.Head.Text = headTxt
			rel.Tail.Text = tailTxt
			allRelations = append(allRelations, rel)
		}
	}

	return entities, allRelations, words, spansInfo, nil
}

func isMeaningfulSpan(s string) bool {
	s = strings.TrimSpace(s)
	if len([]rune(s)) < 2 {
		return false
	}
	hasAlphaNum := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			hasAlphaNum = true
			break
		}
	}
	return hasAlphaNum
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
