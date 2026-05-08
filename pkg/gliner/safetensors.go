package gliner

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

type TensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

type SafetensorsHeader map[string]json.RawMessage

func LoadSafetensors(path string) (map[string]*mat.Dense, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open safetensors file: %w", err)
	}
	defer file.Close()

	// 1. Read the header size (8 bytes, little endian)
	var headerSize uint64
	err = binary.Read(file, binary.LittleEndian, &headerSize)
	if err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	// 2. Read the JSON header
	headerBytes := make([]byte, headerSize)
	_, err = io.ReadFull(file, headerBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON header: %w", err)
	}

	var header SafetensorsHeader
	err = json.Unmarshal(headerBytes, &header)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON header: %w", err)
	}

	// The rest of the file is the data buffer.
	// We read it all into memory for fast slicing.
	dataBuffer, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read data buffer: %w", err)
	}

	tensors := make(map[string]*mat.Dense)

	for key, rawValue := range header {
		if key == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(rawValue, &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", key, err)
		}

		if info.Dtype != "F32" {
			// For GLiNER we expect F32. If other types appear, we'd handle them here.
			return nil, fmt.Errorf("unsupported dtype %s for tensor %s", info.Dtype, key)
		}

		if len(info.DataOffsets) != 2 {
			return nil, fmt.Errorf("invalid data_offsets for tensor %s", key)
		}

		start, end := info.DataOffsets[0], info.DataOffsets[1]
		tensorBytes := dataBuffer[start:end]

		// Convert bytes to []float64 (since gonum mat.Dense uses float64)
		numElements := len(tensorBytes) / 4
		dataF64 := make([]float64, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(tensorBytes[i*4 : (i+1)*4])
			val := math.Float32frombits(bits)
			dataF64[i] = float64(val)
		}

		// Create gonum mat.Dense
		var r, c int
		if len(info.Shape) == 1 {
			r = info.Shape[0]
			c = 1
		} else if len(info.Shape) == 2 {
			r = info.Shape[0]
			c = info.Shape[1]
		} else {
			return nil, fmt.Errorf("unsupported shape length %d for tensor %s", len(info.Shape), key)
		}

		tensors[key] = mat.NewDense(r, c, dataF64)
	}

	return tensors, nil
}
