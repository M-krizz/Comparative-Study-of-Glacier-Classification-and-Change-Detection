import cv2
import numpy as np

image_path = '/Users/padminibalagopal/Desktop/glacier.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}.")
    exit()

original = image.copy()
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])

glacier_mask = cv2.inRange(image, lower_white, upper_white)
non_glacier_mask = cv2.bitwise_not(glacier_mask)

highlighted = original.copy()
highlighted[non_glacier_mask > 0] = [0, 255, 0]

output = cv2.addWeighted(original, 0.7, highlighted, 0.3, 0)

cv2.imshow("Original Image", original)
cv2.imshow("Non-Glacier Mask", non_glacier_mask)
cv2.imshow("Highlighted Non-Glacier Areas", output)
cv2.imwrite('/Users/padminibalagopal/Desktop/non_glacier_highlighted.jpg', output)

cv2.waitKey(0)
cv2.destroyAllWindows()