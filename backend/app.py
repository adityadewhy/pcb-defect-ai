import os
import base64
import io
import gc
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    return "PCB AI Backend is awake and running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    conf_threshold = float(request.form.get('confidence', 0.25))

    try:
        img = Image.open(file.stream).convert('RGB')

        # Run inference in no_grad mode to save memory on 512MB RAM server
        with torch.no_grad():
            results = model.predict(source=img, conf=conf_threshold, imgsz=640)
        
        result = results[0]

        defects = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            confidence = float(box.conf[0])
            defects.append({
                'class_name': class_name,
                'confidence': round(confidence, 4)
            })

        # Draw bounding boxes
        res_plotted = result.plot() 
        annotated_img = Image.fromarray(res_plotted[..., ::-1])  

        buffer = io.BytesIO()
        annotated_img.save(buffer, format='JPEG', quality=85)
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_img}"

        # Clean up memory explicitly
        del results, result, res_plotted, annotated_img, img
        gc.collect()

        return jsonify({
            "success": True,
            "image_url": image_url,
            "defects": defects
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)