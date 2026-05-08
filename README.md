# hugot-gliner2

A 100% native Go implementation of the GLiNER2 relationship extraction pipeline. This project provides zero-dependency (other than ONNX Runtime) end-to-end extraction from raw text strings to structured entity and relation data, utilizing the mathematical equivalence of PyTorch linear projections directly via Gonum.

## Architecture

This pipeline is a complete port of the GLiNER tokenizer, word-pooling, and cross-encoder logic into Go:
- **`processor.go`**: Handles text tokenization and word-to-subword alignment mapping natively without requiring Python.
- **`pipeline.go`**: Abstracted wrapper around `onnxruntime_go` managing the Cross-Encoder context fusion (dynamic prompt generation + extraction of relationship schema embeddings).
- **`math.go`**: Implements span formulation, feed-forward projections, and non-maximum suppression (NMS) using Gonum matrix operations.
- **`types.go`**: Public Go types for `SpanMatch` and `RelationResult`.

For a full technical deep dive into how the matrix operations and cross-encoder prompt mappings work, read [ALGORITHM.md](ALGORITHM.md).

## Project Structure

```
├── cmd/demo/         # Command-line application demonstrating E2E text-to-relations
├── pkg/gliner/       # Core Go pipeline abstractions, math ops, and ONNX interactions
├── tests/            # Automated test suite and JSON validation logic
└── scripts/export/   # Python scripts to export weights, tokenizers, and reference tensors
```

## Running the Demo

1. **Prerequisites**: You must have `libonnxruntime.so` installed on your machine.
2. Edit `cmd/demo/main.go` and ensure the `SetSharedLibraryPath` points to your local ONNX Runtime library file.
3. Run the pipeline:
   ```bash
   go run cmd/demo/main.go
   ```

### Output Example

```
🚀 Initializing GLiNER2 Pipeline Demo (E2E Text-to-Relations)...

📄 Procesando texto original:
El pasado lunes, Elena Rodríguez...

✨ Entidades Candidatas:
   🔹 TechNova Solutions (score: 10.4284)
   🔹 Nordic AI (score: 11.2330)
   🔹 Lukas Virtanen (score: 11.2725)
   ...

🤝 Relaciones Extraídas:
   [Lukas Virtanen] ---> fundó ---> [Nordic AI] (Confianza: H=1.0000, T=0.9999)
   [Elena Rodríguez] ---> trabaja en ---> [TechNova Solutions] (Confianza: H=0.9962, T=0.9910)
```

## Integrating into your Go App

```go
// 1. Initialize Pipeline
pipeline, err := gliner.NewPipeline(
	"encoder.onnx",
	"count_embed.onnx",
	"gliner_classifiers.safetensors",
	"tokenizer.json",
	"prompt_ids.json", // Optional: Custom relations prompt
	nil,
)
defer pipeline.Close()

// 2. Extract directly from text
entities, relations, words, spansInfo, err := pipeline.ExtractFromText("Lukas Virtanen fundó Nordic AI.")
```
