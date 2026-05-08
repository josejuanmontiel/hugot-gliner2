package gliner

// Entity represents an extracted named entity.
type Entity struct {
	ID    int     `json:"id"`
	Text  string  `json:"text"`
	Label string  `json:"label"`
	Start int     `json:"start"`
	End   int     `json:"end"`
	Score float64 `json:"score"`
}

// Relation represents an extracted relationship between two entities.
type Relation struct {
	Head  int     `json:"head"` // ID of the head entity
	Tail  int     `json:"tail"` // ID of the tail entity
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// Result represents the final output of the GLiNER pipeline.
type Result struct {
	Entities  []Entity   `json:"entities"`
	Relations []Relation `json:"relations"`
}
