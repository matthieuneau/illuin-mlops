import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from google.cloud import storage
from torch import jit

from gcpUtils import upload_gcs_model_to_vertex, upload_to_bucket

# Load environment variables
load_dotenv()


# Define a simple model that multiplies input by 2
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.multiplier = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.multiplier


def create_and_save_model(local_path="./models"):
    """Create a dummy model, save it using TorchScript, and return the path"""
    # Create directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Create the model
    model = DummyModel()
    model.eval()  # Set to evaluation mode

    # Test the model
    test_input = torch.tensor([5.0])
    output = model(test_input)
    print(f"Model test: {test_input.item()} * 2 = {output.item()}")

    # Save using torch.save (standard approach)
    torch_save_path = os.path.join(local_path, "dummy_state_dict.pt")
    torch.save(model.state_dict(), torch_save_path)
    print(f"Model state dict saved to {torch_save_path}")

    # Save the entire model
    full_model_path = os.path.join(local_path, "dummy_full.pt")
    torch.save(model, full_model_path)
    print(f"Full model saved to {full_model_path}")

    # Save using TorchScript (better for loading without knowing architecture)
    script_model = jit.script(model)
    script_path = os.path.join(local_path, "dummy_script.pt")
    script_model.save(script_path)
    print(f"TorchScript model saved to {script_path}")


# create_and_save_model()

# upload_to_bucket(
#     bucket_name="dummy-model",
#     source_file_path="./models/dummy_script.pt",
#     destination_blob_name="dummy_script.pt",
# )

# upload_gcs_model_to_vertex("gs://dummy-model/dummy_script.pt", "dummy_model")
