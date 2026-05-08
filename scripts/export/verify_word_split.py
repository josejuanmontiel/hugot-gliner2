from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
record = model.processor.transform_and_format("Texto uno dos tres.", {"relations": []})
print(record.text_word_first_positions)
print(record.text_tokens)
print(model.processor.tokenizer.convert_ids_to_tokens(record.input_ids))
