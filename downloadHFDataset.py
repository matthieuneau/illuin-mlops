import os

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# --- Configuration ---
dataset_name = "HuggingFaceFW/fineweb-edu"
num_samples = 1000
max_tokens = 350  # Should be 512 but for some reason at inference model complains about too long sequences
output_dir = "data/texts"
text_field = "text"  # Assumes the text content is in the 'text' field
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceFW/fineweb-edu-classifier"
)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceFW/fineweb-edu-classifier",
)


os.makedirs(output_dir, exist_ok=True)

# Streaming mode avoids downloading the entire massive dataset.
print(f"Loading dataset '{dataset_name}' in streaming mode...")
dataset = load_dataset(dataset_name, split="train", streaming=True)


print(
    f"Processing the first {num_samples} samples and truncating to {max_tokens} tokens..."
)
count = 0
try:
    # Use dataset.take(num_samples) to get an iterable of the first N samples.
    for i, sample in enumerate(dataset.take(num_samples)):
        # Defensive check in case streaming yields fewer samples than requested.
        if i >= num_samples:
            break

        # Check if the expected text field exists in the sample.
        if text_field not in sample:
            print(
                f"Warning: Sample {i} does not contain the field '{text_field}'. Found fields: {list(sample.keys())}"
            )
            print("Skipping this sample.")
            continue  # Skip this sample if the text field isn't found

        # Extract text content.
        text_content = sample[text_field]

        encoded_input = tokenizer.encode(
            text_content,
            truncation=True,  # Enable truncation
            max_length=max_tokens,  # Set the maximum sequence length
            add_special_tokens=True,  # Usually True for BERT-based tasks
        )
        truncated_text = tokenizer.decode(encoded_input, skip_special_tokens=True)

        # Write text content to the file, using UTF-8 encoding.
        file_path = os.path.join(output_dir, f"{i}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(truncated_text)
            count += 1
            # Print progress update every 100 files.
            if count % 100 == 0:
                print(f"Saved {count}/{num_samples} files...")
        except IOError as e:
            print(f"Error writing file {file_path}: {e}")
            # Consider whether to stop or continue if a file write fails.
            # For example, you might want to `break` here.

except Exception as e:
    # Catch potential errors during dataset loading or iteration.
    print(f"An error occurred while loading or processing the dataset: {e}")

print(f"\nFinished processing.")
print(f"Successfully saved {count} text files to '{output_dir}'.")
