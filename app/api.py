# Create or modify api.py
from flask import Flask, request, jsonify
import torch
from torchvision.models.quantization import resnext101_64x4d, ResNeXt101_64X4D_QuantizedWeights
from torchvision.transforms import ToTensor
from PIL import Image
import io
import requests
from threading import Thread
from ultralytics import YOLO
import os
import numpy as np

app = Flask(__name__)

# Load classification model
weights = ResNeXt101_64X4D_QuantizedWeights.DEFAULT
classification_model = resnext101_64x4d(weights=weights, quantize=True)
classification_model.eval()
preprocess = weights.transforms()

# Load detection model
print("Loading YOLO model...")
detection_model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully")

@app.route('/predict', methods=['POST'])
def predict():
    """Original classification endpoint remains unchanged"""
    if 'image' in request.files:
        # Direct image upload
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = ToTensor()(img)
    elif 'url' in request.json:
        # URL to image
        response = requests.get(request.json['url'])
        img = Image.open(io.BytesIO(response.content))
        img = ToTensor()(img)
    else:
        return jsonify({'error': 'No image data provided'}), 400

    # Process image
    batch = preprocess(img).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        prediction = classification_model(batch).squeeze(0).softmax(0)
        
    # Get results
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    
    return jsonify({
        'class_id': class_id,
        'score': score,
        'category': category_name
    })

@app.route('/detect', methods=['POST'])
def detect():
    """New endpoint for object detection using YOLO"""
    if 'image' in request.files:
        # Direct image upload
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    elif 'url' in request.json:
        # URL to image
        response = requests.get(request.json['url'])
        img_bytes = response.content
        img = Image.open(io.BytesIO(img_bytes))
    else:
        return jsonify({'error': 'No image data provided'}), 400

    # Save temporarily for YOLO
    temp_path = "temp_image.jpg"
    img.save(temp_path)
    
    # Run YOLO detection
    results = detection_model(temp_path)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = detection_model.names[class_id]
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'class': class_name,
                'confidence': confidence
            })
    
    # Clean up temp file
    try:
        os.remove(temp_path)
    except:
        pass
    
    return jsonify({
        'detections': detections,
        'count': len(detections)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Combined endpoint that provides both classification and detection"""
    if 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    elif 'url' in request.json:
        response = requests.get(request.json['url'])
        img_bytes = response.content
        img = Image.open(io.BytesIO(img_bytes))
    else:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Save temporarily for YOLO
    temp_path = "temp_image.jpg"
    img.save(temp_path)
    
    # Classification
    img_tensor = ToTensor()(img)
    batch = preprocess(img_tensor).unsqueeze(0)
    
    with torch.no_grad():
        class_prediction = classification_model(batch).squeeze(0).softmax(0)
    
    class_id = class_prediction.argmax().item()
    score = class_prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    
    # Detection
    detection_results = detection_model(temp_path)
    
    detections = []
    for r in detection_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            det_class_id = int(box.cls[0].item())
            det_class_name = detection_model.names[det_class_id]
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'class': det_class_name,
                'confidence': confidence
            })
    
    # Clean up temp file
    try:
        os.remove(temp_path)
    except:
        pass
    
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
    # Run with multiple threads to handle concurrent requests
    app.run(host='0.0.0.0', port=5000, threaded=True)