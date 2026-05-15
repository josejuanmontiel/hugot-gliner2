//go:build gonum

package math

import (
	stdmath "math"

	"gonum.org/v1/gonum/mat"
)

// toGonum converts our Tensor to a Gonum Dense matrix.
func toGonum(t *Tensor) *mat.Dense {
	r, c := t.Dims()
	dataF64 := make([]float64, len(t.Data))
	for i, v := range t.Data {
		dataF64[i] = float64(v)
	}
	return mat.NewDense(r, c, dataF64)
}

// fromGonum converts a Gonum Dense matrix back to our Tensor.
func fromGonum(m *mat.Dense) *Tensor {
	r, c := m.Dims()
	dataF32 := make([]float32, r*c)
	flat := m.RawMatrix().Data
	for i, v := range flat {
		dataF32[i] = float32(v)
	}
	return &Tensor{
		Data:  dataF32,
		Shape: []int{r, c},
	}
}

// Linear applies Y = X * W^T + B using Gonum.
func Linear(x *Tensor, weight *Tensor, bias *Tensor) *Tensor {
	gX := toGonum(x)
	gW := toGonum(weight)

	r, _ := gX.Dims()
	outC, _ := gW.Dims()

	var y mat.Dense
	y.Mul(gX, gW.T())

	if bias != nil {
		biasVec := bias.Data
		for i := 0; i < r; i++ {
			for j := 0; j < outC; j++ {
				val := y.At(i, j) + float64(biasVec[j])
				y.Set(i, j, val)
			}
		}
	}

	return fromGonum(&y)
}

// ReLU applies the ReLU activation function in-place.
func ReLU(x *Tensor) {
	for i := range x.Data {
		if x.Data[i] < 0 {
			x.Data[i] = 0
		}
	}
}

// Sigmoid applies the Sigmoid activation function in-place.
func Sigmoid(x *Tensor) {
	for i := range x.Data {
		x.Data[i] = 1.0 / (1.0 + float32(stdmath.Exp(float64(-x.Data[i]))))
	}
}

// Dot computes the dot product of two 1D vectors.
func Dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
// GELU applies the Gaussian Error Linear Unit activation.
func GELU(x *Tensor) {
	for i := range x.Data {
		v := float64(x.Data[i])
		x.Data[i] = float32(0.5 * v * (1.0 + stdmath.Tanh(stdmath.Sqrt(2.0/stdmath.Pi)*(v+0.044715*stdmath.Pow(v, 3.0)))))
	}
}

// SiLU (Swish) applies x * sigmoid(x).
func SiLU(x *Tensor) {
	for i := range x.Data {
		v := float64(x.Data[i])
		sig := 1.0 / (1.0 + stdmath.Exp(-v))
		x.Data[i] = float32(v * sig)
	}
}

// Biaffine applies a bilinear transformation. Stub for DSL showcase.
func Biaffine(x1, x2, weight *Tensor) *Tensor {
	// Uses the same native logic for simplicity in this demo
	rows1, cols1 := x1.Dims()
	rows2, cols2 := x2.Dims()
	outData := make([]float32, rows1*rows2)
	for i := 0; i < rows1; i++ {
		for j := 0; j < rows2; j++ {
			var sum float32
			for k := 0; k < cols1; k++ {
				for l := 0; l < cols2; l++ {
					sum += x1.Data[i*cols1+k] * x2.Data[j*cols2+l]
				}
			}
			outData[i*rows2+j] = sum
		}
	}
	return &Tensor{Data: outData, Shape: []int{rows1, rows2}}
}
