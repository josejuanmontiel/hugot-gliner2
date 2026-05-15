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

// 2. Extract directly from text
entities, relations, words, spansInfo, err := pipeline.ExtractFromText("Lukas Virtanen founded Nordic AI.")
```

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
