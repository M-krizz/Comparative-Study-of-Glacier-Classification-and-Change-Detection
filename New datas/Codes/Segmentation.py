import cv2
import numpy as np

# Specify the image path
image_path = "D:/GLOF/New datas/Images/Siachen_Glacier_2020.jpg"

# Load the image
image = cv2.imread(image_path)

# Check if| the image is loaded correctly
if image is None:
    print(f"Error: Unable to load image at {image_path}.")
    exit()

# Convert image to HSV for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the white glacier color range in HSV
lower_white = np.array([0, 0, 180])  # Adjust based on glacier color
upper_white = np.array([180, 50, 255])

# Create the glacier mask
glacier_mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Generate the non-glacier mask
non_glacier_mask = cv2.bitwise_not(glacier_mask)

# Highlight non-glacier areas in green
highlighted = image.copy()
highlighted[non_glacier_mask > 0] = [0, 255, 0]  # Green overlay

# Blend the original and highlighted image
output = cv2.addWeighted(image, 0.7, highlighted, 0.3, 0)

# Convert ground truth to binary (for evaluation)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, ground_truth = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
ground_truth = (ground_truth / 255).astype(np.uint8)  # Normalize 255 → 1

# Convert glacier_mask to binary (0 and 1)
glacier_mask = (glacier_mask / 255).astype(np.uint8)

# Flatten both masks for evaluation
glacier_mask_flat = glacier_mask.flatten()
ground_truth_flat = ground_truth.flatten()

# Calculate Accuracy
accuracy = np.mean(glacier_mask_flat == ground_truth_flat)

# Calculate Intersection over Union (IoU)
intersection = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))
union = np.sum((glacier_mask_flat == 1) | (ground_truth_flat == 1))
iou = intersection / float(union) if union > 0 else 0

# Calculate Precision, Recall, and F1 Score
tp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 1))  # True Positives
fp = np.sum((glacier_mask_flat == 1) & (ground_truth_flat == 0))  # False Positives
fn = np.sum((glacier_mask_flat == 0) & (ground_truth_flat == 1))  # False Negatives

precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print accuracy results
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Glacier Mask", (glacier_mask * 255).astype(np.uint8))  # Convert 1 → 255 for display
cv2.imshow("Highlighted Non-Glacier Areas", output)

# Save the output image
output_path = r'D:/GLOF/New datas/Images/Siachen_Glacier_2020.jpg'
cv2.imwrite(output_path, output)
print(f"Segmented image saved at {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()