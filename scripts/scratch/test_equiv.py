import torch
from gliner2 import GLiNER2

def test_equivalence():
    model_name = "fastino/gliner2-base-v1"
    model = GLiNER2.from_pretrained(model_name)
    model.eval()
    
    count_embed = model.count_embed
    count_embed.eval()
    
    dummy_pc_emb = torch.randn(2, 768)
    
    # Run with count = 2
    out_2 = count_embed(dummy_pc_emb, 2)
    
    # Run with count = 20
    out_20 = count_embed(dummy_pc_emb, 20)
    
    # Compare first 2 elements
    diff = (out_2 - out_20[:2]).abs().max()
    print("Max difference:", diff.item())

if __name__ == "__main__":
    test_equivalence()
