package layers

import "hugot-gliner2/pkg/math"

// Module defines the interface for any tensor transformation.
type Module interface {
	Forward(x *math.Tensor) *math.Tensor
}

// LinearLayer represents a dense layer: Y = XW^T + B.
type LinearLayer struct {
	Weight *math.Tensor
	Bias   *math.Tensor
}

func (l *LinearLayer) Forward(x *math.Tensor) *math.Tensor {
	return math.Linear(x, l.Weight, l.Bias)
}

// ActivationLayer wraps an in-place activation function like ReLU or Sigmoid.
type ActivationLayer struct {
	Fn func(*math.Tensor)
}

func (a *ActivationLayer) Forward(x *math.Tensor) *math.Tensor {
	a.Fn(x)
	return x
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
