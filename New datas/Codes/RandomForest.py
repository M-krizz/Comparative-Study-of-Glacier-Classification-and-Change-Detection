import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, f1_score
from sklearn.model_selection import train_test_split
import sys

# ✅ Fix encoding issue
sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------
# 1️⃣ Load Glacier Images & Generate Masks Automatically
# ---------------------------

IMAGE_SIZE = 256
image_dir = "D:/GLOF/New datas/Images/rf"  
mask_dir = r"output\randomForest_1"

# Check if the image directory exists
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Error: Directory '{image_dir}' does not exist.")

# Create mask directory if it doesn't exist
os.makedirs(mask_dir, exist_ok=True)

# Ensure only image files are selected
glacier_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith((".jpg", ".png"))]

if not glacier_images:
    raise RuntimeError("Error: No valid images found in the directory.")

print(f"Found {len(glacier_images)} glacier images. Generating masks...")

def generate_mask(image):
    """Automatically segments glacier vs. non-glacier using K-Means."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.reshape((-1, 3))  # Flatten image

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  
    kmeans.fit(image_flat)
    segmented = kmeans.labels_.reshape(image.shape[:2])  
    
    return (segmented * 255).astype(np.uint8)  # Convert to binary mask (0, 255)

# Process each glacier image and generate its mask
for img_path in glacier_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read {img_path}, skipping...")
        continue
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = generate_mask(img)  

    # Save mask image
    mask_filename = os.path.join(mask_dir, os.path.basename(img_path).replace(".jpg", "_mask.png").replace(".png", "_mask.png"))
    cv2.imwrite(mask_filename, mask)
    print(f"Saved mask: {mask_filename}")

print("Glacier masks generated successfully!")

# ---------------------------
# 2️⃣ Load Images & Generated Masks for Training
# ---------------------------

def load_dataset(image_dir, mask_dir):
    images, masks = [], []
    
    for file in os.listdir(image_dir):
        if not file.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", "_mask.png").replace(".png", "_mask.png"))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Warning: Missing image or mask for {file}, skipping...")
            continue
        
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Flatten each sample into a vector
        images.append(img.flatten())  
        masks.append(mask.flatten())  

    images = np.array(images)  
    masks = np.array(masks)   

    return images, masks

# Load the dataset
X, Y = load_dataset(image_dir, mask_dir)

if len(X) == 0 or len(Y) == 0:
    raise RuntimeError("Error: No valid images or masks found. Check dataset paths.")

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ---------------------------
# 3️⃣ Train Random Forest for Glacier Segmentation
# ---------------------------

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train) 

# Predict on validation set
rf_pred = rf_model.predict(X_val)

# ✅ Fix ValueError: Convert 255 → 1 for metric calculation
Y_val_flat = (Y_val.reshape(-1) // 255)  # Convert 255 → 1
rf_pred_flat = (rf_pred.reshape(-1) // 255)  # Convert 255 → 1

# Calculate metrics
rf_accuracy = accuracy_score(Y_val_flat, rf_pred_flat)
rf_iou = jaccard_score(Y_val_flat, rf_pred_flat, average='binary')
rf_dice = f1_score(Y_val_flat, rf_pred_flat, average='binary')

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest IoU Score: {rf_iou:.4f}")
print(f"Random Forest Dice Score: {rf_dice:.4f}")

# Print classification report
print(classification_report(Y_val_flat, rf_pred_flat))

# ---------------------------
# 4️⃣ Display Example Results
# ---------------------------

# Display an example prediction
example_idx = 0
if len(X_val) > 0:
    example_img = X_val[example_idx].reshape((IMAGE_SIZE, IMAGE_SIZE))
    example_mask = Y_val[example_idx].reshape((IMAGE_SIZE, IMAGE_SIZE))
    example_pred = rf_pred[example_idx].reshape((IMAGE_SIZE, IMAGE_SIZE))

    # Ensure no NaNs or invalid values
    example_img = np.nan_to_num(example_img)
    example_mask = np.nan_to_num(example_mask)
    example_pred = np.nan_to_num(example_pred)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(example_img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(example_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(example_pred, cmap='gray')

    # ✅ Save the figure to avoid backend conflicts
    output_path = os.path.join(mask_dir, "example_prediction.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Example prediction saved to: {output_path}")
else:
    print("No valid data available for example display.")

print("✅ Glacier segmentation completed successfully!")
