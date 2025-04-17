from transformers import pipeline

# Load the language detection pipeline with the papluca model
lang_detector = pipeline(
    "text-classification", model="papluca/xlm-roberta-base-language-detection"
)

# Classify the language of an input text
result = lang_detector("Bonjour tout le monde")
print(result)
