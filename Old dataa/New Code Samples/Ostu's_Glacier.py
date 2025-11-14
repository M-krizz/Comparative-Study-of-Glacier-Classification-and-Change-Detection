import cv2

image = cv2.imread("D:\\GLOF\\New Code Samples\\Siachen_Glacier_2020.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Otsu’s thresholding
_, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Otsu’s Thresholding 2020", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
