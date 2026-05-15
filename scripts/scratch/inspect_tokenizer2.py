from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
print(model.processor.tokenizer.decode(128002))
print(model.processor.tokenizer.convert_ids_to_tokens(128002))
