import os
import tempfile

import torch
from dotenv import load_dotenv
from google.cloud import storage
from transformers import AutoTokenizer

from gcpUtils import load_model_from_bucket

load_dotenv()

bucket_name = "fineweb_models"
folder_name = "model_init"
file_name = "model_init_script.pt"

client = storage.Client(project="cs-3a-2024-fineweb-mlops")
bucket = client.bucket(bucket_name)

# Construct the blob path using folder_name and file_name
blob_path = f"{folder_name}/{file_name}"
blob = bucket.blob(blob_path)

# Create a named temporary file
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
    temp_path = temp_file.name

try:
    # Download the model to the temporary file
    blob.download_to_filename(temp_path)

    # Load the model
    model = torch.jit.load(temp_path)
    # return model
finally:
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

# model = load_model_from_bucket(
#     bucket_name="fineweb_models",
#     folder_name="model_init",
#     file_name="model_init_script.pt",
# )

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
# # model = torch.jit.load(filepath)

# text = "The American Civil War (April 12, 1861 – May 26, 1865; also known by other names) was a civil war in the United States between the Union[e] ('the North') and the Confederacy ('the South')"
# inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
# del inputs[
#     "token_type_ids"
# ]  # Remove this unexpected when loading the model with torch.jit
# logits = model(**inputs).squeeze()

# score = torch.argmax(logits).item()


# # {"text": "This is a test sentence.", "score": 0.07964489609003067, "int_score": 0}
