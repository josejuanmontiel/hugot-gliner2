# hugot-gliner2

A 100% native Go implementation of the GLiNER2 relationship extraction pipeline. This project provides zero-dependency (other than ONNX Runtime) end-to-end extraction from raw text strings to structured entity and relation data, utilizing a modular mathematical engine with support for both Native Go and Gonum.

## Architecture

This pipeline is a complete port of the GLiNER tokenizer, word-pooling, and cross-encoder logic into Go:
- **`pkg/math`**: Modular mathematical engine handling core tensor operations. Defaults to **Native Go** (zero-dependencies) but supports **Gonum** via build tags (`-tags gonum`).
- **`pkg/gliner`**: Core pipeline logic. Implements PyTorch-like modular layers (`Linear`, `ReLU`, `Sigmoid`) for classification heads.
- **`processor.go`**: Handles text tokenization and word-to-subword alignment mapping natively.
- **`pipeline.go`**: Abstracted wrapper around `onnxruntime_go` managing Cross-Encoder context fusion and relationship schema embeddings.

For a full technical deep dive, read [ALGORITHM.md](ALGORITHM.md).

## Getting Started

### 1. Download Model Assets

The pipeline requires several assets (ONNX models, head weights, and tokenizer) to function. You can download them using the provided script or manually from [Hugging Face](https://huggingface.co/josejuanmontiel/hugot-gliner2-assets):

```bash
chmod +x download_models.sh
./download_models.sh
```

| File | Description |
| :--- | :--- |
| `encoder.onnx` / `.data` | The main Transformer Encoder (Cross-Encoder logic). |
| `count_embed.onnx` / `.data` | Specialized embeddings for position/count representation. |
| `gliner_classifiers.safetensors` | **Modular weights** for Span Projections and Entity Classifiers. |
| `tokenizer_out/tokenizer.json` | Tokenizer configuration and vocabulary. |
| `tests/testdata/prompt_ids.json` | (Optional) Pre-calculated token IDs for the relation schema. |

> [!IMPORTANT]
> The `.onnx.data` files are required for large models. Ensure they are in the same directory as their corresponding `.onnx` files.

### 2. Prerequisites

You must have the ONNX Runtime shared library installed:
- **Linux**: `libonnxruntime.so`
- **macOS**: `libonnxruntime.dylib`
- **Windows**: `onnxruntime.dll`

### 3. Running the Demo

Run the end-to-end demonstration from the project root:
```bash
go run cmd/demo/main.go
```

#### Output Example

```
🚀 Initializing GLiNER2 Pipeline Demo (E2E Text-to-Relations)...

📄 Processing original text:
Last Monday, Elena Rodriguez, Chief Operating Officer of TechNova Solutions...

🧠 Running E2E pipeline (Tokenization -> ONNX -> Modular Math)...

✨ Candidate Entities:
   🔹 TechNova Solutions (score: 0.8306)
   🔹 Nordic AI (score: 0.6706)
   🔹 Lukas Virtanen (score: 0.9806)

🤝 Extracted Relations:
   [Lukas Virtanen] ---> works at ---> [TechNova Solutions] (Confidence: H=0.9806, T=0.9475)
   [TechNova Solutions] ---> acquired ---> [Nordic AI] (Confidence: H=0.9504, T=1.0000)
   [Lukas Virtanen] ---> founded ---> [Nordic AI] (Confidence: H=0.9993, T=0.9795)

✅ Demo Finished.
```

## Project Structure

```
├── cmd/demo/         # Demonstration CLI application
├── pkg/gliner/       # Core Go pipeline abstractions and layers
├── pkg/math/         # Modular mathematical engine (Native/Gonum)
├── tests/            # E2E tests and simulation data
├── scripts/export/   # Essential Python scripts for model conversion
└── scripts/scratch/  # Internal development and inspection utilities
```

## Export Scripts

The `/scripts/export` directory contains the "source of truth" for the Go implementation:
- `export_gliner.py`: Converts the PyTorch model to ONNX and extracts head weights to Safetensors.
- `validate_tensors.py`: Ensures mathematical parity between the Python reference and the Go implementation.

## Integrating into your Go App

```go
import (
	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/ortinit"
)

// 1. Initialize ONNX Runtime (robust platform search)
ortinit.SetupONNX()

// 2. Initialize Pipeline
pipeline, err := gliner.NewPipeline(
	"encoder.onnx",
	"count_embed.onnx",
	"gliner_classifiers.safetensors",
	"tokenizer_out/tokenizer.json",
	"tests/testdata/prompt_ids.json",
	nil, // Custom labels (optional)
)
defer pipeline.Close()

// 3. Extract directly from text
entities, relations, words, spansInfo, err := pipeline.ExtractFromText("Lukas Virtanen founded Nordic AI.")
```

## Fastino-compatible Fiber API

The repository includes a reusable Fiber application in `pkg/server` and a
runnable server in `cmd/api`. It implements the documented GLiNER form of
`POST /v1/chat/completions` for entity extraction, classification, structured
records, relations, and combined schemas. The extraction result is returned as
a JSON string in `choices[0].message.content`.

### Build prerequisites

In addition to ONNX Runtime, `github.com/daulet/tokenizers` requires its native
`libtokenizers` archive. Download the archive matching the version in `go.mod`
and your platform from the
[tokenizers releases](https://github.com/daulet/tokenizers/releases), then make
its directory visible to the external linker:

```bash
export CGO_LDFLAGS="-L/absolute/path/to/libtokenizers"
./download_models.sh
go build -o gliner-api ./cmd/api
```

`download_models.sh` writes assets atomically and verifies their SHA-256
checksums before replacing local files.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADDR` | `:8080` | Listen address |
| `MODEL_ID` | `fastino/gliner2-base-v1` | Required request model ID |
| `API_KEY` | empty | Optional Bearer or `X-API-Key` authentication |
| `ONNXRUNTIME_LIB_PATH` | platform lookup | ONNX Runtime shared library |
| `ENCODER_PATH` | `encoder.onnx` | Encoder model |
| `COUNT_EMBED_PATH` | `count_embed.onnx` | Count embedding model |
| `CLASSIFIERS_PATH` | `gliner_classifiers.safetensors` | Span/classification heads |
| `TOKENIZER_PATH` | `tokenizer_out/tokenizer.json` | Tokenizer configuration |
| `MAX_BODY_BYTES` | `2097152` | Maximum HTTP request size |
| `MAX_TEXT_BYTES` | `262144` | Maximum analyzed text size |
| `INFERENCE_TIMEOUT` | `5m` | Per-request timeout |
| `INFERENCE_QUEUE_DEPTH` | `16` | Maximum admitted in-flight/queued requests |
| `INFERENCE_PROVIDER` | `auto` | `auto`, `cpu`, or Apple-accelerated `coreml` |
| `COREML_COMPUTE_UNITS` | `CPUAndGPU` | `CPUAndGPU`, `CPUAndNeuralEngine`, `ALL`, or `CPUOnly` |
| `COREML_MODEL_FORMAT` | `NeuralNetwork` | Core ML `NeuralNetwork` or `MLProgram` graph format |
| `COREML_CACHE_DIR` | empty | Optional compiled Core ML model cache directory |
| `COREML_PROFILE` | `false` | Log the Core ML hardware compute plan |
| `COREML_LOW_PRECISION` | `false` | Allow low-precision GPU accumulation |

Start the server:

```bash
./gliner-api
```

On Apple silicon, `INFERENCE_PROVIDER=auto` attempts ONNX Runtime's Core ML
execution provider and safely falls back to CPU if it is unavailable. To
require Metal-capable GPU execution and fail startup when Core ML is missing:

```bash
INFERENCE_PROVIDER=coreml \
COREML_COMPUTE_UNITS=CPUAndGPU \
COREML_CACHE_DIR="$PWD/.coreml-cache" \
./gliner-api
```

The ONNX Runtime shared library must be built with Core ML support. This is the
native Apple acceleration path for the Go ONNX pipeline. MLX itself does not
load ONNX graphs and has no maintained Go binding, so this backend does not
introduce a Python MLX subprocess. `GET /readyz` reports the selected execution
provider.

Call it with a Fastino-compatible request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fastino/gliner2-base-v1",
    "messages": [{"role": "user", "content": "Ada Lovelace worked with Charles Babbage."}],
    "schema": {
      "entities": [
        {"name": "person", "description": "a named individual"}
      ],
      "relations": ["worked_with"]
    },
    "threshold": 0.5,
    "include_confidence": true,
    "include_spans": true,
    "store": false
  }'
```

`GET /healthz` reports process liveness and `GET /readyz` confirms that the
model is loaded. Authentication is disabled when `API_KEY` is empty. Streaming,
batch text arrays, inference persistence/history, GLiDE/SystemOne, training,
and GLiNER2.5-only record modes are intentionally not implemented. A request
with `stream: true` receives an OpenAI-style `400` response.

Dynamic schemas are bounded to 64 tasks and 256 total fields/labels so that
oversized prompts are rejected before model inference.

Interactive Swagger documentation is available at
`http://localhost:8080/swagger/`. The embedded OpenAPI 3.0 document is exposed
at `http://localhost:8080/openapi.json` for client generation and tooling.

### Reusing the library

Implement `gliner.Engine` to provide another backend, or pass a
`*gliner.Pipeline` to `server.New`. The server depends only on the interface,
which keeps HTTP contract tests independent from large model assets.

## Manual Model Export (Python to Go)

If you want to export the models yourself (e.g., to use a different GLiNER2 base model), follow these steps:

1. **Install Python dependencies**:
   ```bash
   pip install torch gliner2 safetensors onnx
   ```

2. **Run the export script**:
   ```bash
   python scripts/export/export_gliner.py
   ```
   This will generate:
   - `encoder.onnx` (and its `.data` file if needed)
   - `count_embed.onnx`
   - `gliner_classifiers.safetensors`

3. **Export the tokenizer**:
   ```bash
   python scripts/export/export_tokenizer.py
   ```
   This will create the `tokenizer_out/` directory with `tokenizer.json`.
