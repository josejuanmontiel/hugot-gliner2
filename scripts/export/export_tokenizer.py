from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
model.processor.tokenizer.save_pretrained("tokenizer_out")
