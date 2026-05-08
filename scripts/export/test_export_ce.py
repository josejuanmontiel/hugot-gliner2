import torch
from gliner2 import GLiNER2

def test_export_count_embed():
    model_name = "fastino/gliner2-base-v1"
    model = GLiNER2.from_pretrained(model_name)
    model.eval()
    
    count_embed = model.count_embed
    count_embed.eval()
    
    # Dummy inputs
    # pc_emb: (M, D) - M is number of fields (2 for relations), D is hidden size (768)
    dummy_pc_emb = torch.randn(2, 768)
    # gold_count_val: int or 0-D tensor
    dummy_count = torch.tensor(1, dtype=torch.long)
    
    print("Testing forward pass...")
    out = count_embed(dummy_pc_emb, dummy_count)
    print("Forward pass successful. Output shape:", out.shape)
    
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            count_embed,
            (dummy_pc_emb, dummy_count),
            "count_embed.onnx",
            input_names=["pc_emb", "count"],
            output_names=["struct_proj"],
            dynamic_axes={
                "pc_emb": {0: "num_fields"},
            },
            opset_version=14,
            do_constant_folding=True
        )
        print("Successfully exported count_embed.onnx")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")

if __name__ == "__main__":
    test_export_count_embed()
