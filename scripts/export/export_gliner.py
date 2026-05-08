import torch
from gliner2 import GLiNER2
from safetensors.torch import save_file
import os

def main():
    print("Loading GLiNER2 model...")
    model_name = "fastino/gliner2-base-v1"
    model = GLiNER2.from_pretrained(model_name)
    
    print("\nModel Architecture:")
    print(model)
    
    torch_model = model
    torch_model.eval()
    
    print("\n--- Model inspection ---")
    for name, module in torch_model.named_children():
        print(f"Child: {name}")
    
    # 1. Export the Transformer Backbone to ONNX
    print("\nExporting Transformer Backbone to ONNX...")
    
    # We need to export `torch_model.encoder` which is the DeBERTa backbone.
    encoder = torch_model.encoder
    encoder.eval()
    
    # Create dummy inputs
    dummy_input_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 16, dtype=torch.long)
    
    # ONNX Export
    try:
        torch.onnx.export(
            encoder,
            (dummy_input_ids, dummy_attention_mask),
            "encoder.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print("Successfully exported encoder.onnx")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")
    
    # 2. Export CountEmbed to ONNX
    print("\nExporting CountEmbed to ONNX...")
    class CountEmbedWrapper(torch.nn.Module):
        def __init__(self, count_embed):
            super().__init__()
            self.count_embed = count_embed
            
        def forward(self, pc_emb):
            # Hardcode gold_count_val to max_count (20) for ONNX static export
            return self.count_embed(pc_emb, self.count_embed.max_count)

    count_embed_wrapper = CountEmbedWrapper(torch_model.count_embed)
    count_embed_wrapper.eval()
    
    # Dummy pc_emb (M=2 for relations, D=768)
    dummy_pc_emb = torch.randn(2, 768)
    
    try:
        torch.onnx.export(
            count_embed_wrapper,
            (dummy_pc_emb,),
            "count_embed.onnx",
            input_names=["pc_emb"],
            output_names=["struct_proj"],
            dynamic_axes={
                "pc_emb": {0: "num_fields"},
                "struct_proj": {1: "num_fields"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print("Successfully exported count_embed.onnx")
    except Exception as e:
        print(f"Failed to export CountEmbed ONNX: {e}")

    # 3. Extract weights of linear classifiers
    print("\nExtracting weights to Safetensors...")
    state_dict = torch_model.state_dict()
    
    # We filter out the transformer weights to keep the safetensors small (only classifiers)
    classifier_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("encoder.") and not key.startswith("count_embed."):
            classifier_state_dict[key] = value
            
    print(f"Found {len(classifier_state_dict)} tensors for classifiers/span rep.")
    save_file(classifier_state_dict, "gliner_classifiers.safetensors")
    print("Saved gliner_classifiers.safetensors")

if __name__ == "__main__":
    main()
