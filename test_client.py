import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import time

# Wait a bit for server to start if running in the same container
print("Waiting for Triton server to start...")
time.sleep(10)

# Preprocess your image
img = Image.open("test_image.jpg").resize((224, 224))
img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0

# Normalize with ImageNet stats
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_array = (img_array - mean) / std

try:
    # Set up Triton client
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Wait for server to be ready
    if not client.is_server_ready():
        print("Waiting for server to be ready...")
        while not client.is_server_ready():
            time.sleep(1)
    
    # Create inference input
    inputs = [httpclient.InferInput("input", [1, 3, 224, 224], "FP32")]
    inputs[0].set_data_from_numpy(img_array.reshape(1, 3, 224, 224))
    
    # Create inference output
    outputs = [httpclient.InferRequestedOutput("output")]
    
    # Run inference
    results = client.infer("resnext101", inputs, outputs=outputs)
    
    # Get results
    output_data = results.as_numpy("output")
    class_id = np.argmax(output_data)
    score = output_data[0][class_id]
    
    # Read labels
    with open("model_repository/resnext101/1/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    print(f"Prediction: {labels[class_id]}")
    print(f"Confidence: {score * 100:.2f}%")
    
except Exception as e:
    print(f"Error: {e}")