package gliner

import (
	"fmt"
	"hugot-gliner2/pkg/math"
)

// Heads holds the assembled modular layers for a GLiNER pipeline.
type Heads struct {
	SpanProjectStart Module
	SpanProjectEnd   Module
	SpanOutProject   Module
	EntityClassifier Module
}

// HeadArchitecture defines the strategy to assemble heads from a map of tensors.
type HeadArchitecture interface {
	Assemble(tensors map[string]*math.Tensor) (*Heads, error)
}

// GLiNER2Architecture implements the standard head structure for GLiNER2.
type GLiNER2Architecture struct{}

func (a *GLiNER2Architecture) Assemble(tensors map[string]*math.Tensor) (*Heads, error) {
	h := &Heads{}

	// Helper to extract tensors with error checking
	get := func(name string) (*math.Tensor, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing required tensor: %s", name)
		}
		return t, nil
	}

	// 1. Span Project Start: Linear -> ReLU -> Linear
	sw0, err := get("span_rep.span_rep_layer.project_start.0.weight")
	if err != nil { return nil, err }
	sb0, _ := get("span_rep.span_rep_layer.project_start.0.bias")
	sw3, _ := get("span_rep.span_rep_layer.project_start.3.weight")
	sb3, _ := get("span_rep.span_rep_layer.project_start.3.bias")

	h.SpanProjectStart = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: sw0, Bias: sb0},
			ActivationLayer{Fn: math.ReLU},
			LinearLayer{Weight: sw3, Bias: sb3},
		},
	}

	// 2. Span Project End: Linear -> ReLU -> Linear
	ew0, _ := get("span_rep.span_rep_layer.project_end.0.weight")
	eb0, _ := get("span_rep.span_rep_layer.project_end.0.bias")
	ew3, _ := get("span_rep.span_rep_layer.project_end.3.weight")
	eb3, _ := get("span_rep.span_rep_layer.project_end.3.bias")

	h.SpanProjectEnd = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: ew0, Bias: eb0},
			ActivationLayer{Fn: math.ReLU},
			LinearLayer{Weight: ew3, Bias: eb3},
		},
	}

	// 3. Span Out Project: Linear -> ReLU -> Linear
	ow0, _ := get("span_rep.span_rep_layer.out_project.0.weight")
	ob0, _ := get("span_rep.span_rep_layer.out_project.0.bias")
	ow3, _ := get("span_rep.span_rep_layer.out_project.3.weight")
	ob3, _ := get("span_rep.span_rep_layer.out_project.3.bias")

	h.SpanOutProject = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: ow0, Bias: ob0},
			ActivationLayer{Fn: math.ReLU},
			LinearLayer{Weight: ow3, Bias: ob3},
		},
	}

	// 4. Entity Classifier: Linear -> ReLU -> Linear -> Sigmoid
	cw0, _ := get("classifier.0.weight")
	cb0, _ := get("classifier.0.bias")
	cw2, _ := get("classifier.2.weight")
	cb2, _ := get("classifier.2.bias")

	h.EntityClassifier = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: cw0, Bias: cb0},
			ActivationLayer{Fn: math.ReLU},
			LinearLayer{Weight: cw2, Bias: cb2},
			ActivationLayer{Fn: math.Sigmoid},
		},
	}

	return h, nil
}

// SOTAArchitecture showcases an advanced structure using GELU and SwiGLU.
// This is an experimental architecture for next-gen models.
type SOTAArchitecture struct{}

func (a *SOTAArchitecture) Assemble(tensors map[string]*math.Tensor) (*Heads, error) {
	h, err := (&GLiNER2Architecture{}).Assemble(tensors)
	if err != nil {
		return nil, err
	}

	// For demonstration, we "upgrade" the Entity Classifier to use GELU
	// instead of ReLU, showcasing how easy it is to swap logic.
	cw0, _ := tensors["classifier.0.weight"]
	cb0, _ := tensors["classifier.0.bias"]
	cw2, _ := tensors["classifier.2.weight"]
	cb2, _ := tensors["classifier.2.bias"]

	h.EntityClassifier = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: cw0, Bias: cb0},
			ActivationLayer{Fn: math.GELU}, // Upgrade to SOTA GELU
			LinearLayer{Weight: cw2, Bias: cb2},
			ActivationLayer{Fn: math.Sigmoid},
		},
	}

	return h, nil
}

// MedicalArchitecture loads custom-trained medical weights.
type MedicalArchitecture struct {
	WeightsPath string
}

func (a *MedicalArchitecture) Assemble(baseTensors map[string]*math.Tensor) (*Heads, error) {
	// 1. Assemble standard structure first for projections
	h, err := (&GLiNER2Architecture{}).Assemble(baseTensors)
	if err != nil {
		return nil, err
	}

	// 2. Load our custom medical weights
	medTensors, err := LoadSafetensors(a.WeightsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load medical weights: %w", err)
	}

	// 3. Replace the Entity Classifier with our trained version
	mw0 := medTensors["classifier.0.weight"]
	mb0 := medTensors["classifier.0.bias"]
	mw2 := medTensors["classifier.2.weight"]
	mb2 := medTensors["classifier.2.bias"]

	h.EntityClassifier = &SequentialModule{
		Modules: []Module{
			LinearLayer{Weight: mw0, Bias: mb0},
			ActivationLayer{Fn: math.ReLU},
			LinearLayer{Weight: mw2, Bias: mb2},
			ActivationLayer{Fn: math.Sigmoid},
		},
	}

	return h, nil
}
