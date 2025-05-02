import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import base64
import uuid
from ultralytics import YOLO
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the exact models from cylinder_1.py
MODELS = [
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector10/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector8/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector5/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector4/weights/best.pt",
]

# Print which models are found
for model_path in MODELS:
    if os.path.exists(model_path):
        print(f"Found model: {model_path}")
    else:
        print(f"Model not found: {model_path}")

DB_PATH = 'cylinder_detecto.db'

def save_detection_result(image_filename, count, timestamp):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            count INTEGER,
            timestamp TEXT
        )
    ''')
    c.execute('''
        INSERT INTO detections (image_filename, count, timestamp)
        VALUES (?, ?, ?)
    ''', (image_filename, count, timestamp))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'})
    
    try:
        # Save the uploaded image
        unique_id = str(uuid.uuid4())[:8]
        filename = f"detect_{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run detection using the exact function from cylinder_1.py
        result_path, count = run_detection(filepath)
        
        # Save to database
        save_detection_result(filename, count, datetime.now().isoformat())
        
        # Convert result image to base64
        with open(result_path, 'rb') as f:
            img_str = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'count': count,
            'image': img_str
        })
    
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history', methods=['GET'])
def api_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT image_filename, count, timestamp FROM detections ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    results = [
        {'image_filename': row[0], 'count': row[1], 'timestamp': row[2]}
        for row in rows
    ]
    return jsonify(results)

def run_detection(image_path):
    """
    This function is a direct adaptation of run_ensemble from cylinder_1.py,
    modified only to return a path to the saved image and the count
    """
    # Use exactly the same models as in cylinder_1.py
    models = [
        "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector10/weights/best.pt",
        "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector8/weights/best.pt",
        "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector5/weights/best.pt",
        "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector4/weights/best.pt",
    ]
    
    conf = 0.25  # Same confidence threshold as cylinder_1.py
    iou = 0.45   # Same IOU threshold as cylinder_1.py
    
    # Exactly the same detection code from cylinder_1.py
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    all_boxes = []
    all_scores = []
    all_classes = []

    # Run inference for each model
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        model = YOLO(model_path)
        results = model.predict(image, conf=conf, iou=iou)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                score = float(box.conf[0])
                cls = int(box.cls[0])
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_classes.append(cls)

    # If no detections, handle gracefully
    if len(all_boxes) == 0:
        print("No detections found by any model.")
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{os.path.basename(image_path)}')
        cv2.putText(image, "No cylinders detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(result_path, image)
        return result_path, 0

    # Convert to tensors for NMS
    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)
    classes_tensor = torch.tensor(all_classes)

    # Apply NMS (class-agnostic)
    keep = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou)
    final_boxes = boxes_tensor[keep].numpy()
    final_scores = scores_tensor[keep].numpy()
    final_classes = classes_tensor[keep].numpy()

    # Draw results exactly as in cylinder_1.py
    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        conf = final_scores[i]
        cls = final_classes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f'{i+1}: {conf:.2f}'
        cv2.putText(image, label, (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image and return the path and count
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{os.path.basename(image_path)}')
    
    # Add total count label to the image
    cv2.putText(image, f'Total Cylinders: {len(final_boxes)}',
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(result_path, image)
    print(f"Detection complete. Found {len(final_boxes)} cylinders.")
    
    return result_path, len(final_boxes)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 