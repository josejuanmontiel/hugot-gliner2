from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
schema = {"relations": [{lbl: {"head": "", "tail": ""}} for lbl in labels]}

r1 = model.processor.transform_and_format("Texto uno", schema)
r2 = model.processor.transform_and_format("Este es el texto dos y es diferente", schema)

p1 = r1.input_ids[:r1.input_ids.index(128002)+1]
p2 = r2.input_ids[:r2.input_ids.index(128002)+1]

print(f"Prompt 1 len: {len(p1)}")
print(f"Prompt 2 len: {len(p2)}")
print(f"Equal? {p1 == p2}")

# Let's save the prompt ids to a json file
import json
with open("prompt_ids.json", "w") as f:
    json.dump({"prompt_ids": p1, "labels": labels}, f)
print("Saved prompt_ids.json")
