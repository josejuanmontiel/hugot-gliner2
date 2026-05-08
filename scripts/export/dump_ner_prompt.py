from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
labels = ["Person", "Location", "Concept"]
schema = {"entities": {l: {"id": ""} for l in labels}}
record = model.processor.transform_and_format("Texto", schema)
print(record.input_ids)
