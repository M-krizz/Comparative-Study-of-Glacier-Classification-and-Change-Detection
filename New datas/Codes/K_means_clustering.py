import cv2
import numpy as np

# Specify the image path
image_path = 'D:\\GLOF\\Code\\Siachen_Glacier_2020.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert image to grayscale for ground truth generation
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a ground-truth mask using simple thresholding (convert to 0 and 1)
_, ground_truth = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
ground_truth = (ground_truth / 255).astype(np.uint8)  # Convert 255 → 1

# Convert image to float and flatten
Z = image.reshape((-1, 3))  # Flatten the image
Z = np.float32(Z)

# Define K-Means criteria and apply K-Means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5  # Number of clusters (glacier vs non-glacier)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers back to uint8
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((image.shape))

# Identify the glacier cluster (assumed to be the *brightest* cluster)
glacier_cluster = np.argmax(np.mean(centers, axis=1))  # Cluster with highest brightness

# Convert segmented image into a binary mask
glacier_mask = (labels.flatten() == glacier_cluster).astype(np.uint8)
glacier_mask = glacier_mask.reshape((image.shape[:2]))  # Reshape to original size

# Flatten both masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Calculate Accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)

# Calculate Intersection over Union (IoU)
intersection = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))
union = np.sum((glacier_mask_flat == 1) | (ground_truth_flat == 1))
iou = intersection / float(union) if union > 0 else 0  # Fix for division by zero

# Calculate Precision, Recall, and F1 Score
tp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))  # True Positives
fp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 0))  # False Positives
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 1))  # False Negatives

precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Generated Ground Truth Mask", (ground_truth * 255).astype(np.uint8))  # Convert 1 → 255 for display
cv2.imshow("K-Means Segmentation", segmented_image)
cv2.imshow("Binary Glacier Mask", (glacier_mask * 255).astype(np.uint8))  # Convert 1 → 255 for display
cv2.waitKey(0)
cv2.destroyAllWindows()