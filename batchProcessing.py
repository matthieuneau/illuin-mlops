import os
import time

import ray
from transformers import AutoModelForSequenceClassification, AutoTokenizer

texts = []
max_length = 0
for root, dirs, files in os.walk("data/texts"):
    for file in files:
        with open(os.path.join(root, file), "r") as f:
            content = f.read()
            texts.append(content)
            max_length = max(max_length, len(content.split()))

print(f"Max length: {max_length}")

texts = texts[:500]
# print(texts)

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "HuggingFaceTB/fineweb-edu-classifier"
# )
# model.eval()

# input_tokens = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
# print(input_tokens)
# outputs = model(**input_tokens)
# print(outputs)

# print(**input_tokens)


# Better to create a class to avoid loading the model multiple times
@ray.remote(num_cpus=1, memory=5 * 1024 * 1024 * 1024)
class EduClassifierModel2:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        self.model.eval()

    def run_inference(self, texts: list[str]):
        input_tokens = self.tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True
        )
        outputs = self.model(**input_tokens)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        scores = logits.tolist()
        return scores


actor_handle = EduClassifierModel2.remote()
object_ref = actor_handle.run_inference.remote(texts)
# time.sleep(100)
result = ray.get(object_ref)
print(result)
