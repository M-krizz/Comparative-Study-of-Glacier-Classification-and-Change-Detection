import cv2
import numpy as np

# Specify the image path
image_path = 'D:\\GLOF\\Code\\Siachen_Glacier_2025.jpg'

# Load the grayscale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load the input image from {image_path}")
    exit()

# Create a binary ground-truth mask (convert to 0 and 1)
_, ground_truth = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
ground_truth = (ground_truth / 255).astype(np.uint8)  # Convert 255 → 1

# Apply Adaptive Thresholding (convert to 0 and 1)
glacier_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
glacier_mask = (glacier_mask / 255).astype(np.uint8)  # Convert 255 → 1

# Flatten both masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Calculate Accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)

# Calculate Intersection over Union (IoU) with fix for empty masks
intersection = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))
union = np.sum((glacier_mask_flat == 1) | (ground_truth_flat == 1))
iou = intersection / float(union) if union > 0 else 0  # Fix for division by zero

# Calculate Precision, Recall, and F1 Score with fix for empty values
tp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))  # True Positives
fp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 0))  # False Positives
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 1))  # False Negatives

precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')  # IoU will not be nan anymore
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Generated Ground Truth Mask", (ground_truth * 255).astype(np.uint8))  # Convert 1 → 255 for display
cv2.imshow("Adaptive Thresholding Mask", (glacier_mask * 255).astype(np.uint8))  # Convert 1 → 255 for display
cv2.waitKey(0)
cv2.destroyAllWindows() 