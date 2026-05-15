package gliner

import (
	"fmt"
	stdmath "math"

	"hugot-gliner2/pkg/math"
)

// BuildSpans takes the sequence of token embeddings (SeqLen, 768)
// and returns the span representations for all valid spans up to maxSpanLen.
// Returns a matrix of shape (num_spans, 768) and a list of (start, end) pairs.
func (p *Pipeline) BuildSpans(hiddenStates *math.Tensor, maxSpanLen int) (*math.Tensor, [][2]int) {
	seqLen, _ := hiddenStates.Dims()

	// 1. Project start and end features for all tokens using modular heads
	startFeatures := p.SpanProjectStart.Forward(hiddenStates)
	endFeatures := p.SpanProjectEnd.Forward(hiddenStates)

	// 2. Generate all valid span pairs
	var spans [][2]int
	for i := 0; i < seqLen; i++ {
		for j := i; j < seqLen && j < i+maxSpanLen; j++ {
			spans = append(spans, [2]int{i, j})
		}
	}

	numSpans := len(spans)
	// Each span is a concatenation of start_feature and end_feature -> 1536
	spanConcatData := make([]float32, numSpans*1536)
	for idx, span := range spans {
		startIdx := span[0]
		endIdx := span[1]

		for d := 0; d < 768; d++ {
			valStart := startFeatures.At(startIdx, d)
			if valStart < 0 {
				valStart = 0
			}
			spanConcatData[idx*1536+d] = valStart

			valEnd := endFeatures.At(endIdx, d)
			if valEnd < 0 {
				valEnd = 0
			}
			spanConcatData[idx*1536+768+d] = valEnd
		}
	}
	spanConcat := &math.Tensor{Data: spanConcatData, Shape: []int{numSpans, 1536}}

	// 3. Out project the concatenated features
	finalSpans := p.SpanOutProject.Forward(spanConcat)

	return finalSpans, spans
}

// ClassifyEntities takes the final span representations and applies the Entity Classifier.
// Returns a slice of float64 scores for each span.
func (p *Pipeline) ClassifyEntities(spanReps *math.Tensor) []float64 {
	scoresMat := p.EntityClassifier.Forward(spanReps)

	r, _ := scoresMat.Dims()
	scores := make([]float64, r)
	for i := 0; i < r; i++ {
		scores[i] = float64(scoresMat.At(i, 0))
	}
	return scores
}

// NMS (Non-Maximum Suppression) filters overlapping spans keeping the highest score.
func NMS(spans [][2]int, scores []float64, threshold float64) []int {
	var valid []int
	for i, score := range scores {
		if score < threshold {
			continue
		}

		// Check overlap with already accepted spans
		overlap := false
		for _, vIdx := range valid {
			vSpan := spans[vIdx]
			cSpan := spans[i]
			// Overlap condition: c.start <= v.end AND c.end >= v.start
			if cSpan[0] <= vSpan[1] && cSpan[1] >= vSpan[0] {
				overlap = true
				break
			}
		}
		if !overlap {
			valid = append(valid, i)
		}
	}
	return valid
}

// SpanMatch represents an extracted span index and its confidence score.
type SpanMatch struct {
	Index int
	Score float64
}

// RelationResult represents an extracted relation mathematically.
type RelationResult struct {
	Head  SpanMatch
	Tail  SpanMatch
	Label string
}

// ExtractRelations computes the dot product of spanReps and structProj.
// structProj: flat array from ONNX shape (maxCount, numFields, 768).
func ExtractRelations(spanReps *math.Tensor, spansInfo [][2]int, validIndices []int, structProj []float32, maxCount, numFields, hiddenSize int) []RelationResult {
	var results []RelationResult
	seen := make(map[string]bool)

	numSpans, _ := spanReps.Dims()
	projVec := make([]float32, hiddenSize)

	for inst := 0; inst < maxCount; inst++ {
		bestHead := SpanMatch{Index: -1, Score: stdmath.Inf(-1)}
		bestTail := SpanMatch{Index: -1, Score: stdmath.Inf(-1)}

		for fidx := 0; fidx < 2; fidx++ { // 0: head, 1: tail
			if fidx >= numFields {
				break
			}

			// Extract vector structProj[inst, fidx, :]
			offset := (inst * numFields * hiddenSize) + (fidx * hiddenSize)
			copy(projVec, structProj[offset:offset+hiddenSize])

			// Dot product only with valid spanReps
			bestMatch := SpanMatch{Index: -1, Score: stdmath.Inf(-1)}

			for spanIdx := 0; spanIdx < numSpans; spanIdx++ {
				spanData := spanReps.Data[spanIdx*hiddenSize : (spanIdx+1)*hiddenSize]
				dot := float64(math.Dot(projVec, spanData))

				if dot > bestMatch.Score {
					bestMatch.Score = dot
					bestMatch.Index = spanIdx
				}
			}

			// Revert score to sigmoid for output
			bestMatch.Score = 1.0 / (1.0 + stdmath.Exp(-bestMatch.Score))

			if fidx == 0 {
				bestHead = bestMatch
			} else {
				bestTail = bestMatch
			}
		}

		if bestHead.Index != -1 && bestTail.Index != -1 && bestHead.Score > 0.5 && bestTail.Score > 0.5 {
			key := fmt.Sprintf("%d-%d", bestHead.Index, bestTail.Index)
			if !seen[key] {
				seen[key] = true
				results = append(results, RelationResult{
					Head: bestHead,
					Tail: bestTail,
				})
			}
		}
	}

	return results
}
