import cv2
import numpy as np

# Specify the image path
image_path = 'D:\\GLOF\\New Code Samples\\Siachen_Glacier_2015.jpg'

# Load the grayscale image
image = cv2.imread(image_path, 0)

# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load the input image from {image_path}")
    exit()

# Create a simple mask using a binary threshold (adjust the threshold as necessary)
_, ground_truth = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding on the original image for the glacier mask
glacier_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Flatten both masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Calculate accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)

# Calculate Intersection over Union (IoU)
intersection = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))
union = np.sum((glacier_mask_flat == 1) | (ground_truth_flat == 1))
iou = intersection / float(union)

# Calculate precision, recall, and F1 score
tp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))  # True positives
fp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 0))  # False positives
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 1))  # False negatives

precision = tp / float(tp + fp) if tp + fp != 0 else 0
recall = tp / float(tp + fn) if tp + fn != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Generated Ground Truth Mask", ground_truth)
cv2.imshow("Adaptive Thresholding", glacier_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
