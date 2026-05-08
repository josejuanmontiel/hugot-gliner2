package gliner

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Linear applies a Linear layer: Y = X * W^T + B
// Note: PyTorch weights are (out_features, in_features).
// In our safetensors parser, they are loaded as (out_features, in_features).
// So we need X.Mul(X, W.T()) and then add B.
func Linear(x *mat.Dense, weight *mat.Dense, bias *mat.Dense) *mat.Dense {
	r, _ := x.Dims()
	outC, _ := weight.Dims() // Weight is (out_features, in_features)

	var y mat.Dense
	y.Mul(x, weight.T())

	// Add bias to each row
	biasVec := bias.RawMatrix().Data
	for i := 0; i < r; i++ {
		for j := 0; j < outC; j++ {
			val := y.At(i, j) + biasVec[j]
			y.Set(i, j, val)
		}
	}
	return &y
}

// ReLU applies the ReLU activation function in-place.
func ReLU(x *mat.Dense) {
	r, c := x.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if x.At(i, j) < 0 {
				x.Set(i, j, 0)
			}
		}
	}
}

// Sigmoid applies the Sigmoid activation function in-place.
func Sigmoid(x *mat.Dense) {
	r, c := x.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := 1.0 / (1.0 + math.Exp(-x.At(i, j)))
			x.Set(i, j, val)
		}
	}
}

// BuildSpans takes the sequence of token embeddings (SeqLen, 768)
// and returns the span representations for all valid spans up to maxSpanLen.
// Returns a matrix of shape (num_spans, 768) and a list of (start, end) pairs.
func (p *Pipeline) BuildSpans(hiddenStates *mat.Dense, maxSpanLen int) (*mat.Dense, [][2]int) {
	seqLen, _ := hiddenStates.Dims()

	// 1. Project start and end features for all tokens
	// start_features = Linear(ReLU(Linear(hidden, W0, B0)), W3, B3)
	start1 := Linear(hiddenStates, p.weights.SpanProjectStart0W, p.weights.SpanProjectStart0B)
	ReLU(start1)
	startFeatures := Linear(start1, p.weights.SpanProjectStart3W, p.weights.SpanProjectStart3B)

	end1 := Linear(hiddenStates, p.weights.SpanProjectEnd0W, p.weights.SpanProjectEnd0B)
	ReLU(end1)
	endFeatures := Linear(end1, p.weights.SpanProjectEnd3W, p.weights.SpanProjectEnd3B)

	// 2. Generate all valid span pairs
	var spans [][2]int
	for i := 0; i < seqLen; i++ {
		for j := i; j < seqLen && j < i+maxSpanLen; j++ {
			spans = append(spans, [2]int{i, j})
		}
	}

	numSpans := len(spans)
	// Each span is a concatenation of start_feature and end_feature -> 1536
	// 2. Concatenate the features: for span [start, end], we concat outStart[start] and outEnd[end]
	//    and apply ReLU!
	spanConcat := mat.NewDense(numSpans, 1536, nil)
	for idx, span := range spans {
		startIdx := span[0]
		endIdx := span[1]

		for d := 0; d < 768; d++ {
			// Get start value and apply ReLU if we were doing it in place,
			// but wait, we need to apply ReLU to the concatenated vector.
			// Actually we can just apply ReLU element-wise when setting.
			valStart := startFeatures.At(startIdx, d)
			if valStart < 0 {
				valStart = 0
			}
			spanConcat.Set(idx, d, valStart)

			valEnd := endFeatures.At(endIdx, d)
			if valEnd < 0 {
				valEnd = 0
			}
			spanConcat.Set(idx, d+768, valEnd)
		}
	}

	// 3. Out project the concatenated features
	out1 := Linear(spanConcat, p.weights.SpanOutProject0W, p.weights.SpanOutProject0B)
	ReLU(out1)
	finalSpans := Linear(out1, p.weights.SpanOutProject3W, p.weights.SpanOutProject3B)

	return finalSpans, spans
}

// ClassifyEntities takes the final span representations and applies the Entity Classifier.
// Returns a slice of float64 scores for each span.
func (p *Pipeline) ClassifyEntities(spanReps *mat.Dense) []float64 {
	out1 := Linear(spanReps, p.weights.Classifier0W, p.weights.Classifier0B)
	ReLU(out1)
	scoresMat := Linear(out1, p.weights.Classifier2W, p.weights.Classifier2B)
	Sigmoid(scoresMat)

	r, _ := scoresMat.Dims()
	scores := make([]float64, r)
	for i := 0; i < r; i++ {
		scores[i] = scoresMat.At(i, 0)
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
				// We assume scores are sorted, or we handle it greedily
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
func ExtractRelations(spanReps *mat.Dense, spansInfo [][2]int, validIndices []int, structProj []float32, maxCount, numFields, hiddenSize int) []RelationResult {
	var results []RelationResult
	seen := make(map[string]bool)

	numSpans, _ := spanReps.Dims()
	projVec := mat.NewVecDense(hiddenSize, nil)

	for inst := 0; inst < maxCount; inst++ {
		bestHead := SpanMatch{Index: -1, Score: math.Inf(-1)}
		bestTail := SpanMatch{Index: -1, Score: math.Inf(-1)}

		for fidx := 0; fidx < 2; fidx++ { // 0: head, 1: tail
			if fidx >= numFields {
				break
			}

			// Extract vector structProj[inst, fidx, :]
			offset := (inst * numFields * hiddenSize) + (fidx * hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				projVec.SetVec(i, float64(structProj[offset+i]))
			}

			// Dot product only with valid spanReps (spans that survived NMS for entities)
			bestMatch := SpanMatch{Index: -1, Score: math.Inf(-1)}

			for spanIdx := 0; spanIdx < numSpans; spanIdx++ {
				spanVec := spanReps.RowView(spanIdx)
				dot := mat.Dot(projVec, spanVec)

				if dot > bestMatch.Score {
					bestMatch.Score = dot
					bestMatch.Index = spanIdx
				}
			}

			// Revert score to sigmoid for output
			bestMatch.Score = 1.0 / (1.0 + math.Exp(-bestMatch.Score))

			if fidx == 0 {
				bestHead = bestMatch
			} else {
				bestTail = bestMatch
			}
		}

		// Only add if BOTH head and tail have a score > 0.5
		if bestHead.Index != -1 && bestTail.Index != -1 && bestHead.Score > 0.5 && bestTail.Score > 0.5 {
			// Deduplicate overlapping/exact same relations based on the specific span indices
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
