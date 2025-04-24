import requests
import time
import json
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_classification():
    """Test the classification endpoint"""
    print("\n===== Testing Classification API =====")
    url = "http://localhost:5001/predict"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Classification API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    print(f"Content: {response.json()}")

def test_detection():
    """Test the detection endpoint"""
    print("\n===== Testing Detection API =====")
    url = "http://localhost:5001/detect"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Detection API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    
    # Display results
    result = response.json()
    print(f"Detected {result['count']} objects:")
    for detection in result['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
    
    # Visualize detections
    try:
        visualize_detections('istockphoto-1412238848-612x612.jpg', result['detections'])
    except Exception as e:
        print(f"Error visualizing detections: {e}")

def test_combined():
    """Test the combined endpoint"""
    print("\n===== Testing Combined API =====")
    url = "http://localhost:5001/analyze"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Combined API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    
    # Display results
    result = response.json()
    print(f"Classification: {result['classification']['category']} ({result['classification']['score']:.2f})")
    print(f"Detected {result['object_detection']['count']} objects:")
    for detection in result['object_detection']['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")

def visualize_detections(image_path, detections):
    """Visualize detection results"""
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Draw bounding boxes
    for detection in detections:
        box = detection['box']
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1-5, f"{detection['class']} {detection['confidence']:.2f}", 
                color='red', fontsize=10, backgroundcolor='white')
    
    plt.savefig('detection_result.jpg')
    print("Detection visualization saved to 'detection_result.jpg'")

if __name__ == "__main__":
    try:
        test_classification()
    except Exception as e:
        print(f"Error in classification test: {e}")
    
    try:
        test_detection()
    except Exception as e:
        print(f"Error in detection test: {e}")
    
    try:
        test_combined()
    except Exception as e:
        print(f"Error in combined test: {e}")
