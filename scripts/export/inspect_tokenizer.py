from gliner2 import GLiNER2
import json

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
text = "El pasado lunes, Elena Rodríguez, Directora de Operaciones"
schema = {"relations": [{"trabaja en": {"head": "", "tail": ""}}]}
record = model.processor.transform_and_format(text, schema)

print(f"Input IDs: {record.input_ids}")
print(f"Word start indices: {record.text_word_first_positions}")
print(f"Text tokens: {record.text_tokens}")
