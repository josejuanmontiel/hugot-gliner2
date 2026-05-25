package gliner

import (
	"reflect"
	"testing"
)

func TestNMS(t *testing.T) {
	// Spans: [start, end]
	spans := [][2]int{
		{0, 2}, // span A: indices 0,1,2
		{1, 3}, // span B: indices 1,2,3 (overlaps A)
		{4, 5}, // span C: indices 4,5 (no overlap)
	}

	// Scores for spans
	scores := []float64{0.9, 0.95, 0.8}

	// Threshold is 0.5.
	// B has the highest score (0.95), so it should be processed first and kept.
	// A overlaps with B, so A should be suppressed despite having score > 0.5.
	// C does not overlap with B, so it should be kept.
	// Wait, the current NMS implementation processes in the order of the slice, not sorted by score!
	// Let's verify how it behaves. The current NMS iterates sequentially.
	
	validIndices := NMS(spans, scores, 0.5)

	// Since current NMS processes sequentially: 
	// 1. A (0.9) is kept.
	// 2. A (0.9) overlaps B, so it's suppressed.
	// 3. C (0.8) is kept.
	expected := []int{1, 2}
	if !reflect.DeepEqual(validIndices, expected) {
		t.Errorf("Expected valid indices %v, got %v", expected, validIndices)
	}
}
