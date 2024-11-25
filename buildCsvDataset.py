import pandas as pd
from datasets import load_dataset
from itertools import islice


# Load the dataset in streaming mode
fine_web_ds = load_dataset(
    "HuggingFaceFW/fineweb-edu-llama3-annotations", split="train", streaming=True
)

# Take only the first 1000 samples
num_samples = 1000
sampled_data = list(islice(fine_web_ds, num_samples))

df = pd.DataFrame(sampled_data)
df.to_csv("tiny-dataset.csv", index=False)
