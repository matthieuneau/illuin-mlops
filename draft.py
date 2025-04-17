from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/fineweb-edu-classifier"
)

print(tokenizer)

# text = "This is a test sentence."
# inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
# outputs = model(**inputs)
# logits = outputs.logits.squeeze(-1).float().detach().numpy()
# score = logits.item()
# result = {
#     "text": text,
#     "score": score,
#     "int_score": int(round(max(0, min(score, 5)))),
# }

# print(result)
# # {'text': 'This is a test sentence.', 'score': 0.07964489609003067, 'int_score': 0}
