package types

// SpanMatch represents an extracted span index and its confidence score.
type SpanMatch struct {
	Index int
	Score float64
	Label string
}

// RelationResult represents an extracted relation mathematically.
type RelationResult struct {
	Head  SpanMatch
	Tail  SpanMatch
	Label string
}

// SpanInfo represents the start and end word indices of a span.
type SpanInfo struct {
	Start int
	End   int
}
