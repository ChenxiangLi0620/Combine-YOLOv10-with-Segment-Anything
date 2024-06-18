# Combine-YOLOv10-with-Segment-Anything

```python
from ultralytics import YOLOv10
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
import matplotlib.pyplot as plt

# Load YOLOv10 model
model = YOLOv10("/home/hammond/yolo_v10_SAM_Example/{HOME}/weights/yolov10n.pt")

image_path = "/home/hammond/cat.jpg"
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Predict bounding boxes with YOLOv10
results = model.predict(source=image_path, conf=0.25)
predicted_boxes = results[0].boxes.xyxy

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="/home/hammond/segment-anything/checkpoints/checkpoint.pth").to(device=torch.device('cuda:0'))
mask_predictor = SamPredictor(sam)

# Transform bounding boxes for SAM
transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])

# Set image for SAM and predict masks
mask_predictor.set_image(image_bgr)
masks, scores, logits = mask_predictor.predict_torch(
   boxes=transformed_boxes,
   multimask_output=False,
   point_coords=None,
   point_labels=None
)

# Move masks to CPU and convert to NumPy arrays
masks = [mask.cpu().numpy() for mask in masks]

# Combine masks
final_mask = np.zeros_like(masks[0][0], dtype=np.uint8)  # Adjust to masks[0][0] to remove extra dimension
for mask in masks:
    final_mask = np.bitwise_or(final_mask, mask[0])  # Adjust to mask[0] to remove extra dimension

# Display the image with the combined mask
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.imshow(final_mask, cmap='gray', alpha=0.7)
plt.axis('off')
plt.show()
```
