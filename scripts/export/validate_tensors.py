import torch
from gliner2 import GLiNER2
import json

def main():
    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()

    # Create dummy hidden states (Batch=1, SeqLen=8, HiddenSize=768)
    torch.manual_seed(42) # For reproducibility
    hidden_states = torch.randn(1, 8, 768)

    project_start = model.span_rep.span_rep_layer.project_start
    project_end = model.span_rep.span_rep_layer.project_end
    out_project = model.span_rep.span_rep_layer.out_project
    classifier = model.classifier

    with torch.no_grad():
        start_feat = project_start(hidden_states)
        end_feat = project_end(hidden_states)
        
        # Generate all spans up to max_span_len = 8
        spans = []
        seq_len = 8
        max_span_len = 8
        for i in range(seq_len):
            for j in range(i, min(seq_len, i + max_span_len)):
                spans.append([i, j])
                
        span_reprs = []
        for s, e in spans:
            # Concatenate start features and end features
            span_reprs.append(torch.cat([start_feat[0, s], end_feat[0, e]]))
        
        span_reprs = torch.stack(span_reprs) # Shape: (5, 1536)
        
        # Apply out_project
        final_spans = out_project(span_reprs) # Shape: (5, 768)
        
        # Apply classification logic
        scores_mat = classifier(final_spans)
        scores = torch.sigmoid(scores_mat)

    # Dump all to JSON
    data = {
        "hidden_states": hidden_states[0].tolist(),
        "spans": spans,
        "final_spans": final_spans.tolist(),
        "scores": scores.squeeze(-1).tolist()
    }

    with open("tensors_test.json", "w") as f:
        json.dump(data, f)
    
    print("Generated tensors_test.json successfully.")

if __name__ == "__main__":
    main()
