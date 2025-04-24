# Update test.py
import requests
import time
import concurrent.futures
import json
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

def test_yolo_api():
    print("\n===== Testing YOLO API =====")
    url = "http://localhost:5000/detect"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to YOLO API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    print(f"Detected objects: {len(response.json()['detections'])}")
    for detection in response.json()['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
    
    # Test concurrent requests
    print("\nTesting concurrent YOLO requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda: requests.post(url, files={'image': open('istockphoto-1412238848-612x612.jpg', 'rb')})) for _ in range(3)]
        
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            print(f"Concurrent response: {response.status_code}, {len(response.json()['detections'])} objects")

def test_combined_api():
    print("\n===== Testing Combined API =====")
    url = "http://localhost:5000/analyze"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Combined API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    data = response.json()
    print(f"Classification: {data['classification']['category']} ({data['classification']['score']:.2f})")
    print(f"Object Detection: {data['object_detection']['count']} objects")
    for detection in data['object_detection']['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")

if __name__ == "__main__":
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(10)
    
    try:
        test_flask_api()
    except Exception as e:
        print(f"Error testing Flask API: {e}")
        
    try:
        test_yolo_api()
    except Exception as e:
        print(f"Error testing YOLO API: {e}")
        
    try:
        test_combined_api()
    except Exception as e:
        print(f"Error testing Combined API: {e}")