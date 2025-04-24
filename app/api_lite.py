from flask import Flask, request, jsonify
import io
import requests
from PIL import Image
import os
import random
import time

app = Flask(__name__)

# Mock classification model
CLASSIFICATION_CATEGORIES = ["grocery store", "shopping mall", "department store", "supermarket", "convenience store"]

# Mock detection model
DETECTION_CLASSES = ["person", "bottle", "chair", "backpack", "cell phone", "handbag"]

@app.route('/predict', methods=['POST'])
def predict():
    """Classification endpoint that simulates ResNeXt101"""
    if 'image' in request.files:
        # Direct image upload
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
    elif 'url' in request.json:
        # URL to image
        response = requests.get(request.json['url'])
        img = Image.open(io.BytesIO(response.content))
    else:
        return jsonify({'error': 'No image data provided'}), 400

    # Simulate processing time
    time.sleep(0.5)
    
    # Simulate classification result
    class_id = 582  # Grocery store in ImageNet
    score = 0.45 + random.random() * 0.1
    category_name = "grocery store"
    
    return jsonify({
        'class_id': class_id,
        'score': score,
        'category': category_name
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Object detection endpoint that simulates YOLO"""
    if 'image' in request.files:
        # Direct image upload
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
    elif 'url' in request.json:
        # URL to image
        response = requests.get(request.json['url'])
        img = Image.open(io.BytesIO(response.content))
    else:
        return jsonify({'error': 'No image data provided'}), 400

    # Simulate processing time
    time.sleep(0.8)
    
    # Get image dimensions
    width, height = img.size
    
    # Simulate detection results (5-8 random objects)
    detections = []
    num_objects = random.randint(5, 8)
    
    for _ in range(num_objects):
        # Random object class
        object_class = random.choice(DETECTION_CLASSES)
        
        # Random bounding box
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)
        
        # Random confidence
        confidence = 0.7 + random.random() * 0.25
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'class': object_class,
            'confidence': confidence
        })
    
    return jsonify({
        'detections': detections,
        'count': len(detections)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Combined endpoint that provides both classification and detection"""
    if 'image' in request.files:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
    elif 'url' in request.json:
        response = requests.get(request.json['url'])
        img = Image.open(io.BytesIO(response.content))
    else:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Simulate processing time
    time.sleep(1.2)
    
    # Classification results
    class_id = 582  # Grocery store in ImageNet
    score = 0.45 + random.random() * 0.1
    category_name = "grocery store"
    
    # Detection results
    width, height = img.size
    detections = []
    num_objects = random.randint(5, 8)
    
    for _ in range(num_objects):
        # Random object class
        object_class = random.choice(DETECTION_CLASSES)
        
        # Random bounding box
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)
        
        # Random confidence
        confidence = 0.7 + random.random() * 0.25
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'class': object_class,
            'confidence': confidence
        })
    
    return jsonify({
        'classification': {
            'class_id': class_id,
            'score': score,
            'category': category_name
        },
        'object_detection': {
            'detections': detections,
            'count': len(detections)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
