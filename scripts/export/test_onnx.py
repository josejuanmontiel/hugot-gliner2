import onnxruntime as ort
import json
import torch
import numpy as np

def main():
    # Load JSON
    with open("hugot_sim.json", "r", encoding="utf-8") as f:
        sim = json.load(f)
    
    # Load ONNX
    sess = ort.InferenceSession("count_embed.onnx", providers=['CPUExecutionProvider'])
    
    # Run for first relation
    pc_emb = np.array(sim["pc_embs"][0], dtype=np.float32) # (2, 768)
    print("pc_emb shape:", pc_emb.shape)
    
    out = sess.run(None, {"pc_emb": pc_emb})[0]
    print("ONNX out shape:", out.shape)
    
    # Print norm of output vectors
    print("Norm of inst=0, head:", np.linalg.norm(out[0, 0]))
    print("Norm of inst=0, tail:", np.linalg.norm(out[0, 1]))
    print("Norm of inst=1, head:", np.linalg.norm(out[1, 0]))
    print("Norm of inst=1, tail:", np.linalg.norm(out[1, 1]))
    
    # Load original PyTorch model
    from gliner2 import GLiNER2
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()
    
    with torch.no_grad():
        pt_out = model.count_embed(torch.tensor(pc_emb), 20).numpy()
        
    print("\nPyTorch out shape:", pt_out.shape)
    print("Norm of inst=0, head:", np.linalg.norm(pt_out[0, 0]))
    
    # Let's get actual span_reps
    batch = model.processor.collate_fn_train([(sim["text_tokens"], {})])
    token_embeddings = model.encoder(
        input_ids=torch.tensor([sim["input_ids"]]),
        attention_mask=torch.tensor([sim["attention_mask"]])
    ).last_hidden_state
    
    # Emulate the prompt cutting: we just care about the span_reps for the text
    # In PyTorch, model.span_rep returns the spans for the entire sequence.
    # In my Go code, I just passed the text word token embeddings to BuildSpans.
    # Let's see what the max dot product is for ANY vector in token_embeddings!
    
    # Just dot product token_embeddings with pt_out[0, 0]
    tok_emb = token_embeddings[0].numpy() # (seq_len, 768)
    head_proj = pt_out[0, 0] # (768,)
    
    dot_products = np.dot(tok_emb, head_proj)
    print("Max dot product with encoder hidden states:", np.max(dot_products))
    print("Min dot product with encoder hidden states:", np.min(dot_products))
    
    # What about out_project?
    # PyTorch does an out_projection of the span_reps!
    # Let's out project the tok_emb just as an approximation.
    # We can use model.span_rep_layer.out_project
    concatenated = torch.cat([token_embeddings[0], token_embeddings[0]], dim=-1) # (seq_len, 1536)
    projected = model.span_rep_layer.out_project(concatenated) # (seq_len, 768)
    
    dot_products_proj = np.dot(projected.detach().numpy(), head_proj)
    print("Max dot product with projected span_reps:", np.max(dot_products_proj))
    print("Min dot product with projected span_reps:", np.min(dot_products_proj))
    
    # Just print the max 5 elements
    sorted_dots = np.sort(dot_products_proj.flatten())[::-1]
    print("Top 5 dots:", sorted_dots[:5])

if __name__ == "__main__":
    main()
