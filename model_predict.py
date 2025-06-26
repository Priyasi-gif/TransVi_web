from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("Priyasi/virus_identification_5")
model = AutoModelForSequenceClassification.from_pretrained("Priyasi/virus_identification_5")

# Assuming label mapping
label_map = {0: "Virus A", 1: "Virus B", 2: "Virus C"}  # Replace with actual mapping

def predict_virus(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(predicted_class, "Unknown Virus")
