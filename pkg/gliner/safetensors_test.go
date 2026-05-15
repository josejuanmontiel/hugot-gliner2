package gliner

import (
	"os"
	"testing"
)

func TestLoadSafetensors(t *testing.T) {
	path := "../../gliner_classifiers.safetensors"
	if _, err := os.Stat(path); err != nil {
		t.Skip("Skipping safetensors test: file not found at", path)
	}
	tensors, err := LoadSafetensors(path)
	if err != nil {
		t.Fatalf("Failed to load safetensors: %v", err)
	}

	if len(tensors) == 0 {
		t.Fatalf("Expected some tensors, got 0")
	}

	t.Logf("Successfully loaded %d tensors", len(tensors))
	for name, tensor := range tensors {
		r, c := tensor.Dims()
		t.Logf("Tensor %s: shape (%d, %d)", name, r, c)
	}
}
