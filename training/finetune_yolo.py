import os
from ultralytics import YOLO

# Absolute paths
BASE_DIR = '/workspaces/TechLead-AI-Assignment/training'
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
DATASET_CONFIG = os.path.join(BASE_DIR, 'retail_dataset.yaml')

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, 'retail_yolo'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'runs', 'detect'), exist_ok=True)

# Create dataset configuration
with open(DATASET_CONFIG, 'w') as f:
    f.write(f'''
# YOLOv8 Dataset Configuration for Retail Detection
path: {DATA_DIR}
train: images
val: images

nc: 5
names: 
  0: person
  1: shopping_cart
  2: product
  3: shelf
  4: checkout
''')

# 1. Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model

print("Starting fine-tuning process...")

# 2. Fine-tune on retail dataset
results = model.train(
    data=DATASET_CONFIG,  # Configuration file
    epochs=25,            # Minimal epochs for demonstration
    imgsz=640,            # Standard image size
    batch=4,              # Small batch size
    patience=5,           # Early stopping
    project=BASE_DIR,     # Output directory
    name='retail_yolo'    # Run name
)

print("Fine-tuning complete!")

# 3. Export the fine-tuned model
model.export(format='onnx')  # Export to ONNX format for deployment
print("Model exported to ONNX format")

# 4. Run inference on test image as demonstration
test_image = os.path.join(IMAGES_DIR, 'istockphoto-1412238848-612x612.jpg')
results = model.predict(test_image, save=True, conf=0.25)
print(f"Inference complete. Results saved to {BASE_DIR}/runs/detect/")