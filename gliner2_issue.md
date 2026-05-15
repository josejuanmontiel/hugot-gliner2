# Proposed GitHub Issue for fastino-ai/GLiNER2

**Title**: [Community] Native Go implementation of GLiNER2 pipeline (hugot-gliner2)

**Body**:

Hello GLiNER2 team!

First of all, thank you for releasing GLiNER2. It's a fantastic model that simplifies many Information Extraction tasks into a single efficient pass, and the "CPU-first" approach is exactly what many production environments need.

I wanted to let you know that I've created a **100% native Go implementation** of the GLiNER2 relationship extraction pipeline: **hugot-gliner2** (https://github.com/josejuanmontiel/hugot-gliner2).

This implementation allows running GLiNER2 models (like `fastino/gliner2-base-v1`) directly in Go applications without requiring a Python environment, PyTorch, or complex RPC setups. It handles the full E2E process:

1. **Native Tokenization**: Word-to-subword alignment mapping natively in Go.
2. **ONNX Inference**: Using `onnxruntime_go` for the Cross-Encoder and Count-Embed modules.
3. **Native Math Ops**: Span formulation and feed-forward projections using Gonum (achieving mathematical equivalence with the original PyTorch linear layers).
4. **Relational Extraction**: Full support for schema-based relationship extraction and non-maximum suppression (NMS).

We are successfully using it for high-performance relational data extraction natively in Go. I thought you might find this useful for your community or for users looking for a lightweight, high-performance Go alternative.

Keep up the great work!
