Hi there!

I'm a big fan of hugot and its mission to bring transformer pipelines to the Go ecosystem. I wanted to share a project I've been working on: hugot-gliner2 (https://github.com/josejuanmontiel/hugot-gliner2).

It's a 100% native Go implementation of the GLiNER2 (Unified Schema-Based Information Extraction) relationship extraction pipeline. It currently provides:

    Zero-dependency (besides ONNX Runtime) end-to-end extraction from raw text.
    Native tokenization and word-to-subword alignment (no Python required).
    Mathematical equivalence with PyTorch linear projections using Gonum for classification heads.
    Support for Entity and Relation extraction in a single unified flow.

I noticed that hugot already supports several pipelines like tokenClassification and zeroShotClassification. GLiNER2 fits very well with these use cases but provides a unified schema-driven approach that might be a great addition or a complementary project.

I'm sharing this in case you find it interesting for potential integration into hugot (as a new pipeline type) or simply to let the Go ML community know there's a native way to run these state-of-the-art models.

Thanks for the great work on hugot!



Hi @josejuanmontiel, thank you for opening this issue! I took a cursory look and I do think it would be nice to be able to include gliner in hugot. I have a few questions:

    Is gonum necessary? Currently we don't have it as a dependency, i wonder if the matrix work can be done purely with arrays or do you need the matrix operations from gonum?
    I see you have a model weights struct in the pipeline.go that are loaded from safetensors but I would have thought those would be stored in the onnx model?

---

**Response**:

Hi! Thank you for the feedback. Those are very sharp questions. I've just pushed a major architectural refactor to address exactly those points and align the project with `hugot`'s philosophy:

1. **Is Gonum necessary?**: Not anymore. I've implemented a modular math engine in `pkg/math`. It now defaults to a **pure Go, zero-dependency implementation** for all tensor operations (Linear, ReLU, Sigmoid). I've kept Gonum only as an optional backend that can be enabled via the `-tags gonum` build tag for those who want that extra bit of performance, but it's no longer a requirement.

2. **Why Safetensors instead of ONNX?**: This is a design choice specific to GLiNER2's modularity. While the heavy Transformer backbone is in ONNX for performance, the classification heads are very lightweight. Storing them in `safetensors` allows us to:
    - **Swap schemas easily**: You can change the entity/relation heads without re-exporting the entire multi-GB transformer model.
    - **Maintainability**: It keeps the Go implementation closer to the "Modular Head" architecture of the original research.
    - **Optimized Loading**: Our new `safetensors` loader is extremely memory-efficient and type-safe.

3. **Other Improvements**:
    - **`pkg/layers`**: Decoupled the pipeline logic from the layer definitions (Linear, Sequential, etc.), following a PyTorch-like modular structure.
    - **`pkg/ortinit`**: Added a robust, platform-agnostic ONNX initialization that handles `.so`, `.dll`, and `.dylib` automatically using build tags.

I believe these changes make `hugot-gliner2` a perfect candidate for a new pipeline type in `hugot`, maintaining your "zero-dependency" philosophy. Let me know what you think!


