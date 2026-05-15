package gliner

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	stdmath "math"
	"os"
	"sort"

	"hugot-gliner2/pkg/math"
)

// SaveSafetensors writes a map of tensors to a file in Safetensors format.
func SaveSafetensors(path string, tensors map[string]*math.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 1. Prepare header
	header := make(map[string]interface{})
	offset := uint64(0)
	for name, t := range tensors {
		size := uint64(len(t.Data) * 4)
		header[name] = map[string]interface{}{
			"dtype":     "F32",
			"shape":     t.Shape,
			"data_offsets": []uint64{offset, offset + size},
		}
		offset += size
	}

	headerJSON, _ := json.Marshal(header)
	headerSize := uint64(len(headerJSON))

	// 2. Write header size (uint64, little-endian)
	if err := binary.Write(f, binary.LittleEndian, headerSize); err != nil {
		return err
	}

	// 3. Write header JSON
	if _, err := f.Write(headerJSON); err != nil {
		return err
	}

	// 4. Write data buffer
	for _, name := range sortedKeys(tensors) { // Sort keys to match offsets
		t := tensors[name]
		if err := binary.Write(f, binary.LittleEndian, t.Data); err != nil {
			return err
		}
	}

	return nil
}

func sortedKeys(m map[string]*math.Tensor) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

type TensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

func LoadSafetensors(path string) (map[string]*math.Tensor, error) {
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

	var header map[string]json.RawMessage
	err = json.Unmarshal(headerBytes, &header)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON header: %w", err)
	}

	// The current position is the start of the data buffer
	dataStart, _ := file.Seek(0, io.SeekCurrent)

	tensors := make(map[string]*math.Tensor)

	for key, rawValue := range header {
		if key == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(rawValue, &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", key, err)
		}

		if info.Dtype != "F32" {
			return nil, fmt.Errorf("unsupported dtype %s for tensor %s", info.Dtype, key)
		}

		startOffset := int64(info.DataOffsets[0])
		endOffset := int64(info.DataOffsets[1])
		size := endOffset - startOffset

		// Read the tensor data
		file.Seek(dataStart+startOffset, io.SeekStart)
		tensorBytes := make([]byte, size)
		_, err = io.ReadFull(file, tensorBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to read data for tensor %s: %w", key, err)
		}

		// Convert bytes to []float32
		numElements := int(size / 4)
		dataF32 := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(tensorBytes[i*4 : (i+1)*4])
			dataF32[i] = stdmath.Float32frombits(bits)
		}

		tensors[key] = &math.Tensor{
			Data:  dataF32,
			Shape: info.Shape,
		}
	}

	return tensors, nil
}
