import cv2
import numpy as np

# Specify the image path
image_path = 'F:/GLOF/Code/Siachen_Glacier_2019.jpg'

# Load the grayscale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load the input image from {image_path}")
    exit()

# Create a simple mask using binary thresholding (ground truth)
_, ground_truth = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
ground_truth = (ground_truth / 255).astype(np.uint8)  # Normalize to 0 and 1

# Apply Otsu's Thresholding for glacier segmentation
_, glacier_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
glacier_mask = (glacier_mask / 255).astype(np.uint8)  # Normalize to 0 and 1

# Flatten both masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Calculate Accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)

# Calculate Intersection over Union (IoU)
intersection = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))
union = np.sum((glacier_mask_flat == 1) | (ground_truth_flat == 1))
iou = intersection / float(union) if union != 0 else 0

# Calculate Precision, Recall, and F1 Score
tp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))  # True positives
fp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 0))  # False positives
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 1))  # False negatives

precision = tp / float(tp + fp) if (tp + fp) != 0 else 0
recall = tp / float(tp + fn) if (tp + fn) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
cv2.imwrite('GroundTruth_mask_2019.jpg',(ground_truth * 255).astype(np.uint8))
cv2.imwrite('Thresholding_mask_2019.jpg',(glacier_mask * 255).astype(np.uint8))

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Generated Ground Truth Mask", (ground_truth * 255).astype(np.uint8))  # Convert back to 255 for display
cv2.imshow("Otsu's Thresholding Mask", (glacier_mask * 255).astype(np.uint8))  # Convert back to 255 for display
cv2.waitKey(0)
cv2.destroyAllWindows()