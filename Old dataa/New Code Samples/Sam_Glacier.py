import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import os

# Path to SAM checkpoint
sam_checkpoint = "c:\\Users\\Administrator\\Downloads\\sam_vit_h_4b8939.pth"

# Verify if the checkpoint exists
if not os.path.exists(sam_checkpoint):
    raise FileNotFoundError(f"Checkpoint not found at {sam_checkpoint}. Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load glacier image
image_path = r"D:\GLOF\New Code Samples\Siachen_Glacier_2014.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert image to RGB for SAM
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Provide a point for segmentation
input_point = np.array([[100, 200]])  # Change these coordinates as needed
input_label = np.array([1])

# Perform segmentation
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# Display the segmentation mask
mask_image = (masks[0].astype(np.uint8) * 255)
cv2.imshow("Segmented Glacier", mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
