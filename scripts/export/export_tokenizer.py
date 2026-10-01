from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
required = ["[P]", "[E]", "[C]", "[R]", "[L]", "[SEP_STRUCT]", "[SEP_TEXT]", "[DESCRIPTION]"]
for token in required:
    encoded = model.processor.tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        raise RuntimeError(f"{token} must encode as one special token, got {encoded}")
model.processor.tokenizer.save_pretrained("tokenizer_out")
