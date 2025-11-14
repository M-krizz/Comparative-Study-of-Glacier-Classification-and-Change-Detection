import cv2
import numpy as np

def enhance_image(image_path, save_path):
    """
    Enhances the input image by applying CLAHE, Gaussian Blur, and sharpening.
    :param image_path: Path to the input raw image.
    :param save_path: Path to save the enhanced image.
    """
    # Load the raw image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Save the enhanced image
    cv2.imwrite(save_path, sharpened)
    print(f"Enhanced image saved to: {save_path}")

# Sample use case
enhance_image("/Users/padminibalagopal/Mohana Krishnan/GLOF/ GLOF/Dataset_Images/Siachen_Glacier_2014.jpg", "/Users/padminibalagopal/Mohana Krishnan/GLOF/ GLOF/Proceesed_Image/Siachen_Glacier_2014_enhanced.jpg")

