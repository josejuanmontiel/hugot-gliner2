#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${MODEL_ASSET_BASE_URL:-https://huggingface.co/josejuanmontiel/hugot-gliner2-assets/resolve/main}"

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download() {
  local remote="$1"
  local destination="$2"
  local expected="$3"
  local temporary="${destination}.part"
  mkdir -p "$(dirname "$destination")"
  curl -fL --retry 3 --retry-all-errors -o "$temporary" "${BASE_URL}/${remote}"
  local actual
  actual="$(checksum "$temporary")"
  if [[ "$actual" != "$expected" ]]; then
    echo "checksum mismatch for ${remote}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
  mv "$temporary" "$destination"
}

download encoder.onnx encoder.onnx ab96eab303d09a1c09e0d03082cb918518927c5232ccc84937c86649dea8b3cc
download encoder.onnx.data encoder.onnx.data a8235d6c3df28c0d2043e4928a726e81f123020ae23f5a63a69d9582127e8acc
download count_embed.onnx count_embed.onnx e4672f496c82abcf4d22a2535acacad13c27fdff1f9a15b3587b02150ff6a6a7
download count_embed.onnx.data count_embed.onnx.data 08f538a79af38404e18687b197abc0c73bc3ddb17af31aac828eb92d3ddfdc7c
download gliner_classifiers.safetensors gliner_classifiers.safetensors feda3ec0fbd719b06af8160a3ba4d878112791b6ea769b23a310d9ec8ea001dc
download tokenizer.json tokenizer_out/tokenizer.json 1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16

echo "GLiNER2 model assets downloaded and verified."
