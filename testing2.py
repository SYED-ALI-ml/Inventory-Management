from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
from PIL import Image
import albumentations as A

class CylinderYOLOTrainer:
    def __init__(self):
        # Get the current working directory for correct path handling
        self.base_path = os.getcwd()
        self.dataset_path = os.path.join(self.base_path, 'cylinder_dataset')
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.labels_path = os.path.join(self.dataset_path, 'labels')
        self.train_data = []
        self.zoom_factor = 1.0
        self.zoom_pos = (0, 0)
        
        # Create directories if they don't exist
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.template_size = None
    
    def zoom_image(self, image, factor, center):
        """Zoom into image at specified center point"""
        height, width = image.shape[:2]
        center_x, center_y = center
        
        # Calculate new dimensions
        new_w = int(width / factor)
        new_h = int(height / factor)
        
        # Calculate start points
        start_x = max(center_x - new_w//2, 0)
        start_y = max(center_y - new_h//2, 0)
        
        # Adjust if exceeding image bounds
        if start_x + new_w > width:
            start_x = width - new_w
        if start_y + new_h > height:
            start_y = height - new_h
        
        # Crop and resize
        cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
        return cv2.resize(cropped, (width, height)), (start_x, start_y, new_w, new_h)

    def add_annotation(self, x1, y1, x2, y2):
        # Ensure coordinates are in correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Store box size as template if it's the first box
        if not self.template_size:
            self.template_size = (x_max - x_min, y_max - y_min)
            print(f"Template size set to: {self.template_size}")
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / self.image_width
        y_center = ((y_min + y_max) / 2) / self.image_height
        width = (x_max - x_min) / self.image_width
        height = (y_max - y_min) / self.image_height
        
        # Add annotation
        self.annotations.append({
            'bbox': (x_min, y_min, x_max, y_max),
            'yolo': f"0 {x_center} {y_center} {width} {height}"
        })
        
        # Update display
        self.update_display()

    def prepare_training_data(self, image_path):
        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            return

        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Error: Could not load image {image_path}")
            return

        print(f"Successfully loaded image with size: {self.original_image.shape}")
        
        self.current_image = self.original_image.copy()
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.annotations = []
        self.drawing = False

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
                
                # If shift is held and we have a template, use it
                if flags & cv2.EVENT_FLAG_SHIFTKEY and self.template_size:
                    w, h = self.template_size
                    self.add_annotation(x, y, x + w, y + h)
                    self.drawing = False
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    img_copy = self.current_image.copy()
                    cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                    
                    # Show dimensions
                    width = abs(x - self.ix)
                    height = abs(y - self.iy)
                    cv2.putText(img_copy, f'{width}x{height}', (x+5, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Show template guide if available
                    if self.template_size:
                        w, h = self.template_size
                        cv2.rectangle(img_copy, (self.ix, self.iy),
                                    (self.ix + w, self.iy + h),
                                    (0, 255, 255), 1)
                    
                    cv2.imshow('Annotation', img_copy)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing:
                    self.drawing = False
                    self.add_annotation(self.ix, self.iy, x, y)

        def update_display():
            self.current_image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            for i, ann in enumerate(self.annotations):
                x_min, y_min, x_max, y_max = ann['bbox']
                cv2.rectangle(self.current_image, (x_min, y_min), 
                            (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(self.current_image, str(i+1), (x_min, y_min-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            info_text = f'Cylinders: {len(self.annotations)}'
            if self.template_size:
                info_text += f' | Template: {self.template_size[0]}x{self.template_size[1]}'
            cv2.putText(self.current_image, info_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Annotation', self.current_image)

        self.update_display = update_display

        # Create window and set mouse callback
        cv2.namedWindow('Annotation')
        cv2.setMouseCallback('Annotation', mouse_callback)
        
        # Show initial image
        self.update_display()
        
        print("\nAnnotation Controls:")
        print("- Left click and drag: Draw bounding box")
        print("- Hold Shift + click: Use template size")
        print("- 's': Save annotations")
        print("- 'r': Reset all annotations")
        print("- 'z': Undo last annotation")
        print("- 'q': Quit annotation")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save image and annotations
                image_id = len(os.listdir(self.images_path))
                image_filename = f'cylinder_{image_id}.jpg'
                label_filename = f'cylinder_{image_id}.txt'
                
                cv2.imwrite(os.path.join(self.images_path, image_filename), 
                           self.original_image)
                with open(os.path.join(self.labels_path, label_filename), 'w') as f:
                    f.write('\n'.join(ann['yolo'] for ann in self.annotations))
                
                print(f"Saved {len(self.annotations)} annotations")
            elif key == ord('r'):
                self.annotations = []
                self.template_size = None
                self.update_display()
                print("Reset annotations")
            elif key == ord('z'):
                if self.annotations:
                    self.annotations.pop()
                    self.update_display()
                    print("Removed last annotation")
        
        cv2.destroyAllWindows()

    def train_model(self):
        # Create dataset.yaml with absolute paths
        yaml_content = f"""
path: {self.dataset_path}  # dataset root dir
train: images  # train images (relative to 'path')
val: images  # val images (relative to 'path')

names:
  0: cylinder
"""
        yaml_path = os.path.join(self.base_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Train YOLO model with 200 epochs instead of 100
        self.model.train(
            data=yaml_path,
            epochs=400,  # Increased from 100 to 200
            imgsz=640,
            batch=8,
            name='cylinder_detector'
        )

    def detect(self, image_path):
        results = self.model.predict(
            image_path,
            conf=0.2,
            iou=0.45
        )
        
        image = cv2.imread(image_path)
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = float(box.conf[0])
                cv2.putText(image, f'{i+1}: {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        total_detections = len(results[0].boxes)
        cv2.putText(image, f'Total Cylinders: {total_detections}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('YOLO Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('yolo_detection_result.jpg', image)
        return total_detections

if __name__ == "__main__":
    trainer = CylinderYOLOTrainer()
    
    # Step 1: Prepare training data
    print("Step 1: Annotating training data...")
    trainer.prepare_training_data('test.jpeg')
    
    # Step 2: Train model
    print("\nStep 2: Training YOLO model...")
    trainer.train_model()
    
    # Step 3: Run detection
    print("\nStep 3: Running detection...")
    num_detected = trainer.detect('test.jpeg')
    print(f"Detection complete. Found {num_detected} cylinders.")

transform = A.Compose([
    A.RandomBrightnessContrast(),
    A.HorizontalFlip(),
    A.Rotate(limit=20),
    # Add more augmentations as needed
])