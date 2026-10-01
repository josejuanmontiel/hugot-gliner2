package gliner

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/yalue/onnxruntime_go"
)

type ExecutionProvider string

const (
	ExecutionProviderAuto   ExecutionProvider = "auto"
	ExecutionProviderCPU    ExecutionProvider = "cpu"
	ExecutionProviderCoreML ExecutionProvider = "coreml"
)

type PipelineOptions struct {
	Provider                       ExecutionProvider
	CoreMLComputeUnits             string
	CoreMLModelFormat              string
	CoreMLModelCacheDirectory      string
	CoreMLProfileComputePlan       bool
	CoreMLLowPrecisionAccumulation bool
}

func (o PipelineOptions) normalized() (PipelineOptions, error) {
	if o.Provider == "" {
		o.Provider = ExecutionProviderAuto
	}
	o.Provider = ExecutionProvider(strings.ToLower(string(o.Provider)))
	if o.Provider != ExecutionProviderAuto && o.Provider != ExecutionProviderCPU && o.Provider != ExecutionProviderCoreML {
		return o, fmt.Errorf("unsupported inference provider %q; expected auto, cpu, or coreml", o.Provider)
	}
	if o.CoreMLComputeUnits == "" {
		o.CoreMLComputeUnits = "CPUAndGPU"
	}
	if o.CoreMLModelFormat == "" {
		o.CoreMLModelFormat = "NeuralNetwork"
	}
	if o.CoreMLModelFormat != "NeuralNetwork" && o.CoreMLModelFormat != "MLProgram" {
		return o, fmt.Errorf("unsupported Core ML model format %q", o.CoreMLModelFormat)
	}
	validComputeUnits := map[string]bool{"ALL": true, "CPUOnly": true, "CPUAndGPU": true, "CPUAndNeuralEngine": true}
	if !validComputeUnits[o.CoreMLComputeUnits] {
		return o, fmt.Errorf("unsupported Core ML compute units %q", o.CoreMLComputeUnits)
	}
	return o, nil
}

// NewDynamicPipelineWithOptions creates a dynamic-schema pipeline with an
// optional Apple Core ML execution provider. Core ML uses the GPU and/or
// Neural Engine and transparently falls back to ONNX CPU for unsupported
// graph nodes. Auto mode tries Core ML on macOS and otherwise uses CPU.
func NewDynamicPipelineWithOptions(encoderPath, countPath, safetensorsPath, tokenizerPath string, options PipelineOptions) (*Pipeline, error) {
	options, err := options.normalized()
	if err != nil {
		return nil, err
	}
	if options.Provider == ExecutionProviderCPU || runtime.GOOS != "darwin" && options.Provider == ExecutionProviderAuto {
		return newPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, "", &GLiNER2Architecture{}, nil)
	}
	if runtime.GOOS != "darwin" {
		return nil, fmt.Errorf("Core ML execution provider requires macOS; current platform is %s", runtime.GOOS)
	}

	sessionOptions, err := coreMLSessionOptions(options)
	if err != nil {
		if options.Provider == ExecutionProviderAuto {
			return newPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, "", &GLiNER2Architecture{}, nil)
		}
		return nil, err
	}
	defer sessionOptions.Destroy()

	pipeline, err := newPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, "", &GLiNER2Architecture{}, sessionOptions)
	if err != nil && options.Provider == ExecutionProviderAuto {
		return newPipeline(encoderPath, countPath, safetensorsPath, tokenizerPath, "", &GLiNER2Architecture{}, nil)
	}
	if err != nil {
		return nil, fmt.Errorf("create Core ML pipeline: %w", err)
	}
	pipeline.provider = string(ExecutionProviderCoreML)
	return pipeline, nil
}

func coreMLSessionOptions(options PipelineOptions) (*onnxruntime_go.SessionOptions, error) {
	sessionOptions, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create ONNX session options: %w", err)
	}
	providerOptions := map[string]string{
		"ModelFormat":              options.CoreMLModelFormat,
		"MLComputeUnits":           options.CoreMLComputeUnits,
		"RequireStaticInputShapes": "0",
		"EnableOnSubgraphs":        "0",
	}
	if options.CoreMLModelCacheDirectory != "" {
		providerOptions["ModelCacheDirectory"] = options.CoreMLModelCacheDirectory
	}
	if options.CoreMLProfileComputePlan {
		providerOptions["ProfileComputePlan"] = "1"
	}
	if options.CoreMLLowPrecisionAccumulation {
		providerOptions["AllowLowPrecisionAccumulationOnGPU"] = "1"
	}
	if err := sessionOptions.AppendExecutionProviderCoreMLV2(providerOptions); err != nil {
		sessionOptions.Destroy()
		return nil, fmt.Errorf("enable Core ML execution provider: %w", err)
	}
	return sessionOptions, nil
}
