import json
from gliner2 import GLiNER2

def main():
    print("Cargando modelo GLiNER2...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    
    with open("texto.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    print("\nTexto cargado. Extrayendo entidades y relaciones...")
    
    # Define the labels we want to extract
    entity_labels = ["Persona", "Organización", "Lugar", "Cargo", "Cantidad", "Fecha", "Tecnología"]
    # The advanced version of GLiNER2 might support relation extraction if we pass relation labels, 
    # but the base model might just do standard NER. Let's see what `predict_entities` does.
    
    entities = model.extract_entities(text, entity_labels)
    
    print("\nExtrayendo relaciones...")
    relation_labels = ["trabaja en", "ubicado en", "fundó", "adquirió"]
    relations = model.extract_relations(text, relation_labels)
    
    print(relations)
    
if __name__ == "__main__":
    main()
