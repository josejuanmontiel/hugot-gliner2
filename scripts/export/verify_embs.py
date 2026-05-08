from gliner2 import GLiNER2
import json

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
schema = {"relations": [{lbl: {"head": "", "tail": ""}} for lbl in labels]}

record = model.processor.transform_and_format("Texto uno", schema)
batch = model.processor.collate_fn_train([("Texto uno", schema)])
token_embs = model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
_, schema_embs = model.processor.extract_embeddings_from_batch(token_embs, batch.input_ids, batch)

# Print indices of [R] token (128006)
ids = record.input_ids
r_indices = [i for i, x in enumerate(ids) if x == 128006]
print("Indices of [R]:", r_indices)

# Compare embs
embs = token_embs[0]
for i in range(len(labels)):
    r1_idx = r_indices[i*2]
    r2_idx = r_indices[i*2 + 1]
    
    extracted_e1 = schema_embs[0][0][1+i*2] # Wait, schema_embs structure is complex
    
    # Just print the first value of the manual slice vs python processor output
    e1_manual = embs[r1_idx][0].item()
    e2_manual = embs[r2_idx][0].item()
    print(f"Manual {i}: {e1_manual}, {e2_manual}")

# We need to correctly parse schema_embs to see if they match exactly
# Actually, let's just dump pc_embs from bridge_processor and see if they match embs[r_idx]
import torch
pc_embs = []
for i in range(4): # relations is task 0
    e = torch.stack(schema_embs[0][i])
    pc_emb = e[1:].tolist()
    pc_embs.append(pc_emb)

print("Match 0,0:", pc_embs[0][0][0] == embs[r_indices[0]][0].item())
print("Match 0,1:", pc_embs[0][1][0] == embs[r_indices[1]][0].item())
