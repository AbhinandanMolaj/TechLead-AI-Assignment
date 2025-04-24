import torch
from torchvision.models.quantization import resnext101_64x4d, ResNeXt101_64X4D_QuantizedWeights
from ultralytics import YOLO
import os

# Ensure directory exists
os.makedirs("model_repository/resnext101/1", exist_ok=True)
os.makedirs("model_repository/yolov8/1", exist_ok=True)

# Export ResNeXt101 model
print("Exporting ResNeXt101 model...")
weights = ResNeXt101_64X4D_QuantizedWeights.DEFAULT
model = resnext101_64x4d(weights=weights, quantize=True)
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Export to TorchScript
traced_model = torch.jit.trace(model, example_input)

# Save the model
torch.jit.save(traced_model, "model_repository/resnext101/1/model.pt")
print("ResNeXt101 model exported successfully")

# Export YOLO model
print("Exporting YOLO model...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.export(format="torchscript", imgsz=640)

# Move the exported model to Triton model repository
os.rename("yolov8n.torchscript", "model_repository/yolov8/1/model.pt")
print("YOLO model exported successfully")