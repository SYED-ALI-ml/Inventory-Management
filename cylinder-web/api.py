from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
import base64
import uuid
import datetime
import json
import pymysql
from pymysql.cursors import DictCursor
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# SQLite Database Configuration
app.config['USE_SQLITE'] = True
app.config['SQLITE_DB'] = 'cylinder_detector.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to get database connection
def get_db():
    try:
        import sqlite3
        conn = sqlite3.connect(app.config['SQLITE_DB'])
        
        # Make SQLite connection return dictionaries
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d
        
        conn.row_factory = dict_factory
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise

# Use the same models as in cylinder_1.py
MODELS = [
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector10/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector8/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector5/weights/best.pt",
    "/Users/syedmohammadalijafri/Cylindrical_Object_Detect/runs/detect/cylinder_detector4/weights/best.pt",
]
# Print the paths to debug
for model_path in MODELS:
    if os.path.exists(model_path):
        print(f"Model found: {model_path}")
    else:
        print(f"Model NOT found: {model_path}")

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'})
    
    try:
        # Generate a unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        filename = f"detect_{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image using ensemble detection
        result_path, count = process_image(filepath)
        
        # Convert result image to base64 for sending to frontend
        with open(result_path, 'rb') as f:
            img_str = base64.b64encode(f.read()).decode('utf-8')
        
        # Current timestamp
        timestamp = datetime.datetime.now()
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Store in SQLite database
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO detections (id, count, image_path, original_path, timestamp, models_used) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    unique_id,
                    count,
                    result_path,
                    filepath,
                    formatted_time,
                    json.dumps([m for m in MODELS if os.path.exists(m)])
                )
            )
            conn.commit()
            conn.close()
            print(f"Detection saved to SQLite database with ID: {unique_id}")
        except Exception as e:
            print(f"Error saving to database: {e}")
        
        return jsonify({
            'success': True,
            'count': count,
            'image': img_str,
            'timestamp': timestamp.strftime('%m/%d/%Y, %H:%M:%S'),
            'detection_id': unique_id
        })
    
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

def process_image(image_path):
    """Wrapper to run the ensemble detection and return the result path and count"""
    result_path, count = run_ensemble(MODELS, image_path)
    return result_path, count

def run_ensemble(models, image_path, conf=0.25, iou=0.45):
    """
    This is an exact copy of the run_ensemble function from cylinder_1.py
    The only modification is that it returns the result_path and count
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        all_boxes = []
        all_scores = []
        all_classes = []

        # Run inference for each model
        for model_path in models:
            try:
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
            except Exception as e:
                print(f"Error with model {model_path}: {e}")
                continue

        # If no detections from any model, return empty result
        if len(all_boxes) == 0:
            print("No detections found by any model.")
            # Save the original image as result
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

        # Draw results
        for i, (x1, y1, x2, y2) in enumerate(final_boxes):
            conf = final_scores[i]
            cls = final_classes[i]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{i+1}: {conf:.2f}'
            cv2.putText(image, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add total count
        cv2.putText(image, f'Total Cylinders: {len(final_boxes)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save the result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{os.path.basename(image_path)}')
        cv2.imwrite(result_path, image)
        
        print(f"Ensemble detection complete. Found {len(final_boxes)} objects.")
        return result_path, len(final_boxes)
    
    except Exception as e:
        print(f"Error in run_ensemble: {e}")
        # Create a default result path
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'error_{os.path.basename(image_path)}')
        # Create a simple error image
        error_img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, "Error processing image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_img, str(e), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(result_path, error_img)
        return result_path, 0

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081) 