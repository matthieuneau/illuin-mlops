import torch

model = torch.jit.load("models/classifier.pt")
model.eval()

input = torch.rand(1, 384)
print(input)

output = model(input)

print(output)
