package math

import (
	"testing"
	"math"
)

func TestLinear(t *testing.T) {
	// Simple 2x2 * 2x2 + 2 Linear layer
	x := &Tensor{
		Data:  []float32{1, 2, 3, 4},
		Shape: []int{2, 2},
	}
	w := &Tensor{
		Data:  []float32{0.5, 0.1, 0.2, 0.3},
		Shape: []int{2, 2},
	}
	b := &Tensor{
		Data:  []float32{0.1, 0.2},
		Shape: []int{2},
	}

	// Y = X * W^T + B
	// W^T = [0.5, 0.2]
	//       [0.1, 0.3]
	// Y[0,0] = 1*0.5 + 2*0.1 + 0.1 = 0.5 + 0.2 + 0.1 = 0.8
	// Y[0,1] = 1*0.2 + 2*0.3 + 0.2 = 0.2 + 0.6 + 0.2 = 1.0
	// Y[1,0] = 3*0.5 + 4*0.1 + 0.1 = 1.5 + 0.4 + 0.1 = 2.0
	// Y[1,1] = 3*0.2 + 4*0.3 + 0.2 = 0.6 + 1.2 + 0.2 = 2.0

	expected := []float32{0.8, 1.0, 2.0, 2.0}
	out := Linear(x, w, b)

	for i := range expected {
		if math.Abs(float64(out.Data[i]-expected[i])) > 1e-5 {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], out.Data[i])
		}
	}
}

func TestReLU(t *testing.T) {
	x := &Tensor{
		Data:  []float32{-1, 0, 1, -2, 5},
		Shape: []int{5},
	}
	expected := []float32{0, 0, 1, 0, 5}
	ReLU(x)

	for i := range expected {
		if x.Data[i] != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], x.Data[i])
		}
	}
}

func TestSigmoid(t *testing.T) {
	x := &Tensor{
		Data:  []float32{0},
		Shape: []int{1},
	}
	// sigmoid(0) = 0.5
	Sigmoid(x)
	if math.Abs(float64(x.Data[0]-0.5)) > 1e-5 {
		t.Errorf("Expected 0.5, got %f", x.Data[0])
	}
}
