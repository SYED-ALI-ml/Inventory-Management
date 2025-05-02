from ultralytics import YOLO
import cv2
import numpy as np
import torch

def run_ensemble(models, image_path, conf=0.25, iou=0.45):
    image = cv2.imread(image_path)
    all_boxes = []
    all_scores = []
    all_classes = []

    # Run inference for each model
    for model_path in models:
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

    # Convert to tensors for NMS
    if len(all_boxes) == 0:
        print("No detections found by any model.")
        return

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
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show and save
    cv2.imshow('Ensemble Detection', image)
    cv2.imwrite('ensemble_detection_result.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Ensemble detection complete. Found {len(final_boxes)} objects.")
    print(f"Total number of detected cylinders: {len(final_boxes)}")

if __name__ == "__main__":
    models = [
        "runs/detect/cylinder_detector10/weights/best.pt",
        "runs/detect/cylinder_detector8/weights/best.pt",
        "runs/detect/cylinder_detector5/weights/best.pt",
        "runs/detect/cylinder_detector4/weights/best.pt",
    ]
    image_path = "test.jpeg"
    run_ensemble(models, image_path)