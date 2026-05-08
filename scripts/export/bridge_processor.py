import json
import torch
from gliner2 import GLiNER2

def main():
    print("Cargando modelo...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    
    with open("texto.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    relation_labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
    
    # We use the actual relations schema format
    schema = {"relations": [{lbl: {"head": "", "tail": ""}} for lbl in relation_labels]}
    
    record = model.processor.transform_and_format(text, schema)
    
    # record is a TransformedRecord
    print("Record generado!")
    
    input_ids = record.input_ids
    mapped_indices = record.mapped_indices
    text_word_first_positions = record.text_word_first_positions
    text_tokens = record.text_tokens
    
    # Get pc_emb for the relations task
    # First, run the encoder to get token_embeddings
    input_ids_tensor = torch.tensor(record.input_ids).unsqueeze(0)
    attention_mask_tensor = torch.tensor(record.attention_mask).unsqueeze(0) if hasattr(record, 'attention_mask') else torch.ones_like(input_ids_tensor)
    
    token_embeddings = model.encoder(
        input_ids=input_ids_tensor,
        attention_mask=attention_mask_tensor
    ).last_hidden_state
    
    # Extract embs using the collator (we have to mock a batch)
    batch = model.processor.collate_fn_train([(text, schema)])
    token_embeddings_batch = model.encoder(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask
    ).last_hidden_state
    
    _, all_schema_embs = model.processor.extract_embeddings_from_batch(
        token_embeddings_batch, batch.input_ids, batch
    )
    
    schema_embs = all_schema_embs[0]
    task_types = batch.task_types[0]
    
    pc_embs = []
    labels = []
    pc_embs = []
    labels = []
    schema_tokens_list = batch.schema_tokens_list[0]
    for i, task in enumerate(task_types):
        if task == "relations":
            embs = torch.stack(schema_embs[i])
            pc_emb = embs[1:].tolist() # Skip [P] token, shape is (2, 768)
            pc_embs.append(pc_emb)
            # The prompt token is the relation label
            labels.append(schema_tokens_list[i][2])
            
    data = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "text_word_indices": text_word_first_positions,
        "text_tokens": text_tokens,
        "mapped_indices": mapped_indices,
        "pc_embs": pc_embs,
        "relation_labels": labels
    }
    
    with open("hugot_sim.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Datos exportados a hugot_sim.json")

if __name__ == "__main__":
    main()
