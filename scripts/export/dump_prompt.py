from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
schema = {"relations": [{lbl: {"head": "", "tail": ""}} for lbl in labels]}
record = model.processor.transform_and_format("Texto de prueba.", schema)
print("Input IDs:", record.input_ids)
print("Decoded:")
for idx in record.input_ids:
    print(f"{idx}: {model.processor.tokenizer.decode([idx])}")
