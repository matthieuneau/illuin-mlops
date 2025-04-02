import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer

# Load the registered model from the MLflow registry (using version 1)
model = mlflow.pytorch.load_model("models:/FinewebEduClassifier/1")

# Load the tokenizer from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

# Prepare an example text and tokenize it
text = "This is a test sentence."
inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)

# Run inference using the loaded model
outputs = model(**inputs)
logits = outputs.logits.squeeze(-1).float().detach().numpy()
score = logits.item()

# Create a result dictionary
result = {
    "text": text,
    "score": score,
    "int_score": int(round(max(0, min(score, 5)))),
}

print(result)
