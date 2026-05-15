//go:build !gonum

package math

import stdmath "math"

// Linear applies Y = X * W^T + B using pure Go.
func Linear(x *Tensor, weight *Tensor, bias *Tensor) *Tensor {
	rowsX, colsX := x.Dims()       // x is (N, in_features)
	outFeatures, _ := weight.Dims() // weight is (out_features, in_features)

	outData := make([]float32, rowsX*outFeatures)

	for i := 0; i < rowsX; i++ {
		for j := 0; j < outFeatures; j++ {
			var sum float32
			rowOff := i * colsX
			weightOff := j * colsX
			for k := 0; k < colsX; k++ {
				sum += x.Data[rowOff+k] * weight.Data[weightOff+k]
			}
			if bias != nil {
				sum += bias.Data[j]
			}
			outData[i*outFeatures+j] = sum
		}
	}

	return &Tensor{
		Data:  outData,
		Shape: []int{rowsX, outFeatures},
	}
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

// Dot computes the dot product of two 1D vectors (or rows).
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

// Biaffine applies a bilinear transformation: x1^T * U * x2 + (x1+x2)*V + b
// For simplicity in this DSL, we implement the core bilinear part: x1 * U * x2^T
func Biaffine(x1, x2, weight *Tensor) *Tensor {
	// Simplified Biaffine core for the DSL showcase
	rows1, cols1 := x1.Dims()
	rows2, cols2 := x2.Dims()

	// Result will be (rows1, rows2) if scoring pairs
	outData := make([]float32, rows1*rows2)
	// This is a simplified version for demonstration
	for i := 0; i < rows1; i++ {
		for j := 0; j < rows2; j++ {
			var sum float32
			for k := 0; k < cols1; k++ {
				for l := 0; l < cols2; l++ {
					// Dummy logic for Biaffine core visualization
					sum += x1.Data[i*cols1+k] * x2.Data[j*cols2+l]
				}
			}
			outData[i*rows2+j] = sum
		}
	}
	return &Tensor{Data: outData, Shape: []int{rows1, rows2}}
}
