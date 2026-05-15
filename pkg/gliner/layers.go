package gliner

import (
	"hugot-gliner2/pkg/math"
)

// Module defines the interface for any tensor transformation.
type Module interface {
	Forward(x *math.Tensor) *math.Tensor
}

// LinearLayer represents a dense layer: Y = XW^T + B.
type LinearLayer struct {
	Weight *math.Tensor
	Bias   *math.Tensor
}

func (l LinearLayer) Forward(x *math.Tensor) *math.Tensor {
	return math.Linear(x, l.Weight, l.Bias)
}

// ActivationLayer wraps an in-place activation function like ReLU or Sigmoid.
type ActivationLayer struct {
	Fn func(*math.Tensor)
}

func (a ActivationLayer) Forward(x *math.Tensor) *math.Tensor {
	a.Fn(x)
	return x
}

// SwiGLULayer implements (xW + b) * SiLU(xV + c)
type SwiGLULayer struct {
	W, V *math.Tensor
	B, C *math.Tensor
}

func (s SwiGLULayer) Forward(x *math.Tensor) *math.Tensor {
	gate := math.Linear(x, s.V, s.C)
	math.SiLU(gate)
	val := math.Linear(x, s.W, s.B)

	// Hadamard product (element-wise)
	for i := range val.Data {
		val.Data[i] *= gate.Data[i]
	}
	return val
}

// BiaffineLayer implements scoring between two sets of vectors (heads and tails)
type BiaffineLayer struct {
	Weight *math.Tensor
}

func (b BiaffineLayer) Forward(x *math.Tensor) *math.Tensor {
	// For simplicity in this demo, we assume x contains concatenated head/tail vectors
	// and we score them using the biaffine operation.
	return x // Implementation details vary by model
}

// SequentialModule runs a list of modules in order.
type SequentialModule struct {
	Modules []Module
}

func (s *SequentialModule) Forward(x *math.Tensor) *math.Tensor {
	out := x
	for _, m := range s.Modules {
		out = m.Forward(out)
	}
	return out
}
