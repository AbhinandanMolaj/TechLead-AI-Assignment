from flask import Flask, request, jsonify
import torch
from torchvision.models.quantization import resnext101_64x4d, ResNeXt101_64X4D_QuantizedWeights
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from PIL import Image
import io
import requests
from threading import Thread

app = Flask(__name__)

# Load model
weights = ResNeXt101_64X4D_QuantizedWeights.DEFAULT
model = resnext101_64x4d(weights=weights, quantize=True)
model.eval()
preprocess = weights.transforms()

@app.route('/predict', methods=['POST'])
def predict():
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
        prediction = model(batch).squeeze(0).softmax(0)
        
    # Get results
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    
    return jsonify({
        'class_id': class_id,
        'score': score,
        'category': category_name
    })

if __name__ == '__main__':
    # Run with multiple threads to handle concurrent requests
    app.run(host='0.0.0.0', port=5000, threaded=True)