# GLiNER2 E2E Extraction Algorithm (Go-Native)

This document details the internal algorithm implemented in `pkg/gliner` to perform joint entity and relation extraction using ONNX and a modular mathematical engine in Go.

## 1. Modular Mathematical Engine (`pkg/math`)

The pipeline uses a switchable "backend" architecture for classification operations:

1. **Native Go (Default)**: Pure Go implementation, no external dependencies, optimized for portability and ease of integration.
2. **Gonum (Optional)**: Optimized implementation using the `gonum` library, activated via build tags (`-tags gonum`).

## 2. Tokenization and Word Pooling (`processor.go`)

Unlike standard Transformer models that associate predictions with subwords (BPE tokens), GLiNER requires extraction to be performed at the **full word level**.

1. **Word Splitting**: The original text is split into words using the regex `[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`. This separates letters/numbers from punctuation marks while maintaining word integrity.
2. **BPE Tokenization**: The entire text is tokenized using the HuggingFace `tokenizer.json`, generating a list of subword IDs and their corresponding strings (e.g., `▁Tech` and `Nova`).
3. **Alignment**: We iterate simultaneously over the list of clean words and the list of BPE subwords. Subwords are accumulated until the original word is reconstructed.
4. **Index Extraction**: We store the index of the *first* subword of each grouped word. This vector (`text_word_indices`) is crucial for the pooling step.

## 3. Cross-Encoder Prompting (`pipeline.go`)

GLiNER is a *Cross-Encoder* model for relation extraction. This means the internal representation of the classes to be predicted (the schema) changes depending on the text context.

1. **Static Prompt**: The IDs of the tokens corresponding to the relation schema are pre-calculated (e.g., `[P] works at ( [R] head [R] tail )`).
2. **Concatenation**: We build a combined input sequence: `[Prompt IDs] + [SEP_TEXT] + [Text IDs]`.
3. **ONNX Inference**: We pass this long sequence through the Transformer model (Encoder). Due to the self-attention mechanism, text words pay attention to target relations, and relation tokens (`[R]`) accumulate semantic information about the text.

## 4. Representation Extraction (`ExtractFromText`)

The model returns a large tensor (`last_hidden_state`). We split it into two parts:

- **Relation Embeddings (`pc_embs`)**: We look for the positions of the `[R]` tokens within the prompt and extract their vectors (dimension 768). These vectors now mathematically represent the concepts of *source* (head) and *target* (tail) for each relation label.
- **Word Embeddings**: Using our `text_word_indices` vector, we jump to the position of the first subword of each word in the tensor and extract its representation (Word Pooling).

## 5. Extraction Mathematics (`pkg/math` and `layers.go`)

From the word embeddings, we use the modular math engine to reproduce the exact PyTorch mathematics:

### Entity Extraction
1. **Span Construction**: We take sliding windows of up to `max_span_len=12` words, concatenating the embedding of the start and end word of each window.
2. **Projection (Feed Forward)**: We pass each span through the classification linear network (whose weights are loaded from `gliner_classifiers.safetensors`).
3. **NMS (Non-Maximum Suppression)**: We filter overlapping spans, retaining only those with the highest score (logit > 0.5).

### Relation Extraction
1. We obtain the empirical representations of each candidate entity.
2. We project these representations into a shared semantic space using matrix operations.
3. We perform the **Dot Product** between the representation of the projected entity and the `pc_embs` extracted in step 4.
4. If the score exceeds the threshold, we confirm the `(Head, Tail)` pair as a valid relation for that class.
