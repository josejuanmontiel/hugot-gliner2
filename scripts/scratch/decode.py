import json
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    with open("hugot_sim.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    ids = data["input_ids"]
    print("Decoded:")
    print(tokenizer.decode(ids))
    print("Relation labels:", data.get("relation_labels", "NOT FOUND"))
    print("PC Embs length:", len(data.get("pc_embs", [])))

if __name__ == "__main__":
    main()
