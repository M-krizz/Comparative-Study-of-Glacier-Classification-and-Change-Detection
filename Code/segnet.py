import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Preprocessing function
def preprocess_image(image_path, input_shape):
    image = load_img(image_path, target_size=input_shape[:2])
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Decoding function to convert predictions into visual format
def decode_segmentation_mask(mask, num_classes):
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    mask = np.argmax(mask, axis=-1)  # Class index for each pixel
    color_mask = colors[mask]
    return color_mask

# Analyze glacier growth across multiple years
def analyze_glacier_growth(image_paths, years, model, input_shape, num_classes, glacier_class=0):
    glacier_areas = []  # To store glacier areas for each year
    
    for year, image_path in zip(years, image_paths):
        # Preprocess image
        img = preprocess_image(image_path, input_shape)
        
        # Predict segmentation mask
        mask = model.predict(img)[0]
        
        # Decode mask for visualization
        decoded_mask = decode_segmentation_mask(mask, num_classes)
        
        # Calculate glacier area (pixels belonging to glacier class)
        glacier_area = np.sum(np.argmax(mask, axis=-1) == glacier_class)
        glacier_areas.append((year, glacier_area))
        
        # Display segmentation result
        plt.figure(figsize=(5, 5))
        plt.title(f"Segmentation for Year {year}")
        plt.imshow(decoded_mask)
        plt.axis('off')
        plt.show()
    
    # Calculate percentage change in glacier area year-over-year
    print("\nYearly Glacier Area Changes:")
    for i in range(1, len(glacier_areas)):
        prev_year, prev_area = glacier_areas[i - 1]
        curr_year, curr_area = glacier_areas[i]
        change_percentage = ((curr_area - prev_area) / prev_area) * 100
        print(f"{prev_year} to {curr_year}: {change_percentage:.2f}% change")
    
    # Plot glacier area over time
    years = [x[0] for x in glacier_areas]
    areas = [x[1] for x in glacier_areas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, areas, marker='o', label="Glacier Area")
    plt.xlabel("Year")
    plt.ylabel("Glacier Area (pixels)")
    plt.title("Glacier Growth Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
image_paths = [
    'Siachen_Glacier_2014.jpg',
    'Siachen_Glacier_2015.jpg',
    'Siachen_Glacier_2016.jpg',
    'Siachen_Glacier_2017.jpg',
    'Siachen_Glacier_2018.jpg',
    'Siachen_Glacier_2019.jpg'
]

years = [2014, 2015, 2016, 2017, 2018, 2019]

# Parameters
input_shape = (256, 256, 3)  # Match your model's input size
num_classes = 3  # Example: glacier, non-glacier, water
glacier_class = 0  # Index for the glacier class

# Analyze glacier growth
analyze_glacier_growth(image_paths, years, segnet_model, input_shape, num_classes, glacier_class)
