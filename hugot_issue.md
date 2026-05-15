# Proposed GitHub Issue for knights-analytics/hugot

**Title**: [Showcase/Request] GLiNER2 Pipeline implementation in Go (hugot-gliner2)

**Body**:

Hi there!

I'm a big fan of `hugot` and its mission to bring transformer pipelines to the Go ecosystem. I wanted to share a project I've been working on: **hugot-gliner2** (https://github.com/josejuanmontiel/hugot-gliner2).

It's a 100% native Go implementation of the **GLiNER2** (Unified Schema-Based Information Extraction) relationship extraction pipeline. It currently provides:

- **Zero-dependency** (besides ONNX Runtime) end-to-end extraction from raw text.
- **Modular Math Engine**: Pure Go implementation for tensor operations (Linear, ReLU, Sigmoid) with optional Gonum support via build tags.
- **Clean Architecture**: Decoupled layers and model heads for high maintainability.
- **Robust ONNX Init**: Multi-platform library discovery (`pkg/ortinit`) that works across Linux, Windows, and macOS.
- **Memory Efficient**: Stream-based `safetensors` loading using `float32` for a minimal RAM footprint.

I've recently completed a major architectural refactor to align the project with `hugot`'s philosophy:
1. **`pkg/math`**: Decouples tensor math. No more mandatory Gonum dependency.
2. **`pkg/layers`**: Functional PyTorch-like layers for classification heads.
3. **`pkg/ortinit`**: idiomatic Go build-tags for platform-specific library loading.

I noticed that `hugot` already supports several pipelines like `tokenClassification`. GLiNER2 fits very well with these use cases but provides a unified schema-driven approach that might be a great addition or a complementary project.

I'm sharing this in case you find it interesting for potential integration into `hugot` (as a new pipeline type) or simply to let the Go ML community know there's a native way to run these state-of-the-art models.

Thanks for the great work on `hugot`!