import requests
import time
import json
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def save_json_result(result, filename='output_results.json'):
    """Save JSON result to a file"""
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Full path for the output file
    output_path = os.path.join('output', filename)
    
    # Save JSON result
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"JSON results saved to {output_path}")

def test_classification():
    """Test the classification endpoint"""
    print("\n===== Testing Classification API =====")
    url = "http://localhost:5001/predict"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Classification API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    
    result = response.json()
    print(f"Content: {result}")
    
    # Save classification result
    save_json_result(result, 'classification_result.json')

def test_detection():
    """Test the detection endpoint"""
    print("\n===== Testing Detection API =====")
    url = "http://localhost:5001/detect"
    
    # Test with direct file upload
    files = {'image': open('istockphoto-1412238848-612x612.jpg', 'rb')}
    print("Sending request to Detection API...")
    response = requests.post(url, files=files)
    print(f"Response: {response.status_code}")
    
    # Display and save results
    result = response.json()
    print(f"Detected {result['count']} objects:")
    for detection in result['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
    
    # Save detection result
    save_json_result(result, 'detection_result.json')
    
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
    
    # Display and save results
    result = response.json()
    print(f"Classification: {result['classification']['category']} ({result['classification']['score']:.2f})")
    print(f"Detected {result['object_detection']['count']} objects:")
    for detection in result['object_detection']['detections']:
        print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
    
    # Save combined result
    save_json_result(result, 'combined_result.json')
    
    # Visualize combined results
    visualize_combined_detections('istockphoto-1412238848-612x612.jpg', result)

def visualize_combined_detections(image_path, result):
    """Visualize combined detection and classification results"""
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Draw bounding boxes
    for detection in result['object_detection']['detections']:
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
    
    # Add classification label
    classification = result['classification']
    plt.title(f"Scene: {classification['category']} (Score: {classification['score']:.2f})")
    
    # Save in output directory
    plt.savefig('output/combined_result.jpg')
    plt.close()  # Close the figure to free up memory
    print("Combined visualization saved to 'output/combined_result.jpg'")

def visualize_detections(image_path, detections):
    """Visualize detection results"""
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
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
    
    # Save in output directory
    plt.savefig('output/detection_result.jpg')
    plt.close()  # Close the figure to free up memory
    print("Detection visualization saved to 'output/detection_result.jpg'")

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