from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # CHANGED: Increased confidence from 0.1 to 0.25 to stop false guesses
    results = model.predict(source=filepath, conf=0.10, imgsz=800)
    
    # Extract defect data
    defects = []
    for r in results:
        for box in r.boxes:
            defects.append({
                "class_name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "coordinates": box.xyxy[0].tolist() 
            })
            
    # NEW: Have YOLO draw the boxes and convert it to a string for the frontend
    result_img = results[0].plot() # This draws the boxes on the image!
    _, buffer = cv2.imencode('.jpg', result_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
            
    os.remove(filepath)
            
    return jsonify({
        "defects": defects, 
        "defect_count": len(defects),
        "image_with_boxes": encoded_image # Send the drawn image back
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)