import torch
from gliner2 import GLiNER2
import json

def main():
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()
    
    with open("texto.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    relation_labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
    
    # We will hook into _find_spans to see what the scores are!
    original_find_spans = model._find_spans
    
    def hooked_find_spans(scores, threshold, text_len, text, start_map, end_map):
        print(f"Hook called! Max score in this relation task: {scores.max().item():.4f}")
        # Find score for first token (which is 'según')
        # Wait, starts=0, widths=0 is the first token
        print(f"Score for start=0, width=0: {scores[0, 0].item():.4f}")
        
        # Let's see the score for 'TechNova Solutions' which has some start and width
        return original_find_spans(scores, threshold, text_len, text, start_map, end_map)
        
    model._find_spans = hooked_find_spans
    
    print("Running extract_relations...")
    relations = model.extract_relations(text, relation_labels)
    print(relations)

if __name__ == "__main__":
    main()
