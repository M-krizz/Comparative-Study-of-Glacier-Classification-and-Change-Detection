import cv2
import numpy as np

# Load grayscale image
image_path = 'F:/GLOF/Code/Siachen_Glacier_2019.jpg'
image = cv2.imread(image_path, 0)

if image is None:
    print(f"Error: Unable to load the input image from {image_path}")
    exit()

# Generate ground truth using binary threshold
_, ground_truth = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Generate glacier mask using adaptive thresholding
glacier_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

# Flatten masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)*100

# Use 255 for binary mask comparison
tp = np.sum((glacier_mask_flat == 255) & (ground_truth_flat == 255))
fp = np.sum((glacier_mask_flat == 255) & (ground_truth_flat == 0))
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 255))

# Intersection over Union
intersection = tp
union = tp + fp + fn
iou = intersection / float(union)*100 if union != 0 else 0

# Precision, Recall, F1
precision = tp / float(tp + fp)*100 if (tp + fp) != 0 else 0
recall = tp / float(tp + fn)*100 if (tp + fn) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Output results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')    
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save and show results
cv2.imwrite('Adaptive_thresholding_GroundTruth_2019.jpg', ground_truth)
cv2.imwrite('Adaptive_thresholding_Mask_2019.jpg', glacier_mask)

cv2.imshow("Original Image", image)
cv2.imshow("Ground Truth Mask", ground_truth)
cv2.imshow("Adaptive Thresholding Mask", glacier_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
