import torch
from torchvision.models.quantization import resnext101_64x4d, ResNeXt101_64X4D_QuantizedWeights
import os

# Ensure directory exists
os.makedirs("model_repository/resnext101/1", exist_ok=True)

# Load model
weights = ResNeXt101_64X4D_QuantizedWeights.DEFAULT
model = resnext101_64x4d(weights=weights, quantize=True)
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Export to TorchScript
traced_model = torch.jit.trace(model, example_input)

# Save the model
torch.jit.save(traced_model, "model_repository/resnext101/1/model.pt")

print("Model exported successfully")