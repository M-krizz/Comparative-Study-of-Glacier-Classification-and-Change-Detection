import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("D:\\GLOF\\New Code Samples\\Siachen_Glacier_2019.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_flat = image.reshape((-1, 3))

# Apply K-Means
k = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(np.float32(image_flat), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to 8-bit and reshape
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape(image.shape)

# Display result
plt.imshow(segmented_image)
plt.title("K-Means Segmentation 2020")
plt.axis("off")
plt.show()
