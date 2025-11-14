import cv2
import numpy as np

# Load the image
image = cv2.imread('D:\GLOF\Code\Siachen_Glacier_2020.jpg')
Z = image.reshape((-1, 3))  # Flatten the image
Z = np.float32(Z)

# Define criteria and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5  # Number of clusters (glacier vs non-glacier)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to uint8 and reshape to the original image shape
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((image.shape))

# Save or display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()