import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing function: Convert to grayscale and apply adaptive thresholding
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))  # Resize to a fixed size
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
    return thresholded

# Function to compare images and highlight glacier changes
def compare_images_for_changes(image_path1, image_path2, year1, year2):
    # Preprocess images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    # Compute absolute difference
    change_mask = cv2.absdiff(img1, img2)

    # Apply threshold to highlight significant changes
    _, change_mask = cv2.threshold(change_mask, 30, 255, cv2.THRESH_BINARY)

    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Thresholded {year1}")
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Thresholded {year2}")
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Changes: {year1} to {year2}")
    plt.imshow(change_mask, cmap='hot')  # Hot colormap highlights changes
    plt.axis('off')
    
    plt.show()

    # Calculate percentage of changes
    total_pixels = change_mask.size
    changed_pixels = np.count_nonzero(change_mask)
    change_percentage = (changed_pixels / total_pixels) * 100
    print(f"Percentage of changed pixels from {year1} to {year2}: {change_percentage:.2f}%")
    
    return change_mask

# Example usage
image_paths = [
    'Siachen_Glacier_2014.jpg',
    'Siachen_Glacier_2015.jpg',
    'Siachen_Glacier_2016.jpg',
    'Siachen_Glacier_2017.jpg',
    'Siachen_Glacier_2018.jpg',
    'Siachen_Glacier_2019.jpg',
    'Siachen_Glacier_2020.jpg'
]

years = [2014, 2015, 2016, 2017, 2018, 2019,2020]

# Compare consecutive years for changes
for i in range(len(image_paths) - 1):
    compare_images_for_changes(image_paths[i], image_paths[i + 1], years[i], years[i + 1])
