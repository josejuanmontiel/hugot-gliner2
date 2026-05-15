package math

// Tensor is our agnostic data structure for weights and activations.
// It uses float32 to save memory and match the original model's precision.
type Tensor struct {
	Data  []float32
	Shape []int
}

// NewTensor creates a new tensor with the given shape.
func NewTensor(shape []int, data []float32) *Tensor {
	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

// At returns the value at (r, c). Assumes 2D tensor.
func (t *Tensor) At(r, c int) float32 {
	cols := t.Shape[1]
	return t.Data[r*cols+c]
}

// Set sets the value at (r, c). Assumes 2D tensor.
func (t *Tensor) Set(r, c int, val float32) {
	cols := t.Shape[1]
	t.Data[r*cols+c] = val
}

// Dims returns the dimensions of the tensor.
func (t *Tensor) Dims() (int, int) {
	if len(t.Shape) == 1 {
		return t.Shape[0], 1
	}
	return t.Shape[0], t.Shape[1]
}
