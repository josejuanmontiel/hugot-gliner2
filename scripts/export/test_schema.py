import torch
from gliner2 import GLiNER2
import json

def main():
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    
    with open("texto.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    relation_labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
    schema = {"relations": [{lbl: {"head": "", "tail": ""}} for lbl in relation_labels]}
    
    # Process
    batch = model.processor.collate_fn_train([(text, schema)])
    
    # Run encoder
    token_embeddings = model.encoder(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask
    ).last_hidden_state
    
    # Extract embs
    all_token_embs, all_schema_embs = model.processor.extract_embeddings_from_batch(
        token_embeddings, batch.input_ids, batch
    )
    
    # For sample 0
    schema_embs = all_schema_embs[0]
    schema_tokens_list = batch.schema_tokens_list[0]
    task_types = batch.task_types[0]
    
    print(f"Number of schemas: {len(schema_embs)}")
    for i, (tokens, task) in enumerate(zip(schema_tokens_list, task_types)):
        embs = torch.stack(schema_embs[i])
        print(f"Task: {task}")
        print(f"Tokens: {tokens}")
        print(f"Embs shape: {embs.shape}")
        
if __name__ == "__main__":
    main()
