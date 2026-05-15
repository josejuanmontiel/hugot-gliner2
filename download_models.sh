#!/bin/bash
# Script to download GLiNER2 model assets for hugot-gliner2

BASE_URL="https://huggingface.co/josejuanmontiel/hugot-gliner2-assets/resolve/main"

echo "📥 Downloading GLiNER2 model assets from Hugging Face..."

# ONNX Models (Core Encoder)
echo "   > encoder.onnx..."
curl -L -O "${BASE_URL}/encoder.onnx"
curl -L -O "${BASE_URL}/encoder.onnx.data"

# ONNX Models (Count Embeddings)
echo "   > count_embed.onnx..."
curl -L -O "${BASE_URL}/count_embed.onnx"
curl -L -O "${BASE_URL}/count_embed.onnx.data"

# Modular Classification Weights
echo "   > gliner_classifiers.safetensors..."
curl -L -O "${BASE_URL}/gliner_classifiers.safetensors"

# Tokenizer Configuration
echo "   > tokenizer.json..."
mkdir -p tokenizer_out
curl -L -o tokenizer_out/tokenizer.json "${BASE_URL}/tokenizer.json"

echo "✅ Done! All assets are in place. You can now run the demo."
