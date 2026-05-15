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
