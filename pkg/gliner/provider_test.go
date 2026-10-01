package gliner

import "testing"

func TestPipelineOptionsNormalization(t *testing.T) {
	options, err := (PipelineOptions{}).normalized()
	if err != nil {
		t.Fatal(err)
	}
	if options.Provider != ExecutionProviderAuto || options.CoreMLComputeUnits != "CPUAndGPU" || options.CoreMLModelFormat != "NeuralNetwork" {
		t.Fatalf("unexpected defaults: %#v", options)
	}

	if _, err := (PipelineOptions{Provider: "cuda"}).normalized(); err == nil {
		t.Fatal("expected invalid provider error")
	}
	if _, err := (PipelineOptions{CoreMLComputeUnits: "GPUOnly"}).normalized(); err == nil {
		t.Fatal("expected invalid Core ML compute units error")
	}
	if _, err := (PipelineOptions{CoreMLModelFormat: "Graph"}).normalized(); err == nil {
		t.Fatal("expected invalid Core ML model format error")
	}
}
