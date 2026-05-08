from gliner2 import GLiNER2
import json

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
schema = {"relations": [{"trabaja en": {"head": "", "tail": ""}}]}

def get_embs(text):
    record = model.processor.transform_and_format(text, schema)
    batch = model.processor.collate_fn_train([(text, schema)])
    token_embs = model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
    _, schema_embs = model.processor.extract_embeddings_from_batch(token_embs, batch.input_ids, batch)
    return schema_embs[0][0][1].tolist() # First element

e1 = get_embs("Texto de prueba uno.")
e2 = get_embs("Completamente distinto texto.")
print(f"Equal? {e1 == e2}")
print(f"Diff: {abs(e1[0] - e2[0])}")
