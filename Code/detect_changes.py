import cv2
import numpy as np

def detect_changes(image1_path, image2_path, save_path):
    """
    Detects changes between two segmented images and calculates the difference.
    :param image1_path: Path to the first segmented image (e.g., 2013).
    :param image2_path: Path to the second segmented image (e.g., 2023).
    :param save_path: Path to save the difference image.
    :return: Areas of the first and second images, and the change in area.
    """
    # Load segmented images
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    # Find the absolute difference
    difference = cv2.absdiff(img1, img2)

    # Calculate the area (number of white pixels)
    area1 = np.sum(img1 == 255)
    area2 = np.sum(img2 == 255)
    area_change = area2 - area1

    # Save the difference image
    cv2.imwrite(save_path, difference)
    print(f"Difference image saved to: {save_path}")

    return area1, area2, area_change

# Sample use case
area_2013, area_2023, area_change = detect_changes(
    "D:\GLOF\Code\Siachen_Glacier_2014.jpg", 
    "D:\GLOF\Code\Siachen_Glacier_2015.jpg", 
    "/Users/padminibalagopal/Mohana Krishnan/GLOF/ GLOF/Proceesed_Image/glacier_difference.jpg"
)

print(f"Glacier Area in 2013: {area_2013} pixels")
print(f"Glacier Area in 2023: {area_2023} pixels")
print(f"Change in Glacier Area: {area_change} pixels")
