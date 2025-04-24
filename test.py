import requests
import time
import concurrent.futures
import json
import sys
import tritonclient.http as httpclient
import numpy as np
from PIL import Image

def test_flask_api():
    print("\n===== Testing Flask API =====")
    url = "http://localhost:5000/predict"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Flask API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    print(f"Content: {response.json()}")
    
    # Test concurrent requests
    print("\nTesting concurrent requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda: requests.post(url, files={'image': open('istockphoto-1412238848-612x612.jpg', 'rb')})) for _ in range(5)]
        
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            print(f"Concurrent response: {response.status_code}, {response.json()['category']}")

def test_triton_api():
    print("\n===== Testing Triton API =====")
    
    try:
        # Preprocess image
        img = Image.open('istockphoto-1412238848-612x612.jpg').resize((224, 224))
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        
        # Create client
        print("Connecting to Triton server...")
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Create inference input
        inputs = [httpclient.InferInput("input", [1, 3, 224, 224], "FP32")]
        inputs[0].set_data_from_numpy(img_array.reshape(1, 3, 224, 224))
        
        # Create inference output
        outputs = [httpclient.InferRequestedOutput("output")]
        
        # Run inference
        print("Sending request to Triton server...")
        results = client.infer("resnext101", inputs, outputs=outputs)
        
        # Get results
        output_data = results.as_numpy("output")
        class_id = np.argmax(output_data[0])
        score = output_data[0][class_id]
        
        print(f"Class ID: {class_id}, Score: {score:.4f}")
        
    except Exception as e:
        print(f"Error testing Triton: {e}")

if __name__ == "__main__":
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(10)
    
    try:
        test_flask_api()
    except Exception as e:
        print(f"Error testing Flask API: {e}")
        
    try:
        test_triton_api()
    except Exception as e:
        print(f"Error testing Triton API: {e}")