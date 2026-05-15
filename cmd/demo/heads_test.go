package main

import (
	"testing"
	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/math"
)

func TestHeadsIntegration(t *testing.T) {
	p := &gliner.Pipeline{
		Heads: &gliner.Heads{},
	}

	// Create weights
	w1 := &math.Tensor{Data: make([]float32, 768*1536), Shape: []int{768, 1536}}
	b1 := &math.Tensor{Data: make([]float32, 768), Shape: []int{768}}
	w2 := &math.Tensor{Data: make([]float32, 1*768), Shape: []int{1, 768}}
	b2 := &math.Tensor{Data: make([]float32, 1), Shape: []int{1}}

	for i := range w1.Data { w1.Data[i] = 0.001 }
	for i := range w2.Data { w2.Data[i] = 0.001 }

	// Create layers explicitly as values (using gliner package types)
	l1 := gliner.LinearLayer{Weight: w1, Bias: b1}
	act1 := gliner.ActivationLayer{Fn: math.ReLU}
	l2 := gliner.LinearLayer{Weight: w2, Bias: b2}
	act2 := gliner.ActivationLayer{Fn: math.Sigmoid}

	p.Heads.EntityClassifier = &gliner.SequentialModule{
		Modules: []gliner.Module{l1, act1, l2, act2},
	}

	projW := &math.Tensor{Data: make([]float32, 768*768), Shape: []int{768, 768}}
	for i := 0; i < 768; i++ { projW.Data[i*768+i] = 1.0 }
	
	projB := &math.Tensor{Data: make([]float32, 768), Shape: []int{768}}
	
	p.Heads.SpanProjectStart = gliner.LinearLayer{Weight: projW, Bias: projB}
	p.Heads.SpanProjectEnd = gliner.LinearLayer{Weight: projW, Bias: projB}
	
	outW := &math.Tensor{Data: make([]float32, 768*1536), Shape: []int{768, 1536}}
	p.Heads.SpanOutProject = gliner.LinearLayer{Weight: outW, Bias: projB}

	wordMat := &math.Tensor{
		Data:  make([]float32, 3*768),
		Shape: []int{3, 768},
	}

	spanReps, spansInfo := p.BuildSpans(wordMat, 2)
	if len(spansInfo) != 5 {
		t.Errorf("Expected 5 spans, got %d", len(spansInfo))
	}

	scores := p.ClassifyEntities(spanReps)
	if len(scores) != 5 {
		t.Errorf("Expected 5 scores, got %d", len(scores))
	}

	for _, s := range scores {
		if s < 0 || s > 1 {
			t.Errorf("Score out of range [0, 1]: %f", s)
		}
	}
}
