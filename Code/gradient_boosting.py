import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load Glacier Images & Generate Masks Automatically
# ---------------------------

IMAGE_SIZE = 256
image_dir = "D:\\GLOF\\Code"  
mask_dir = "D:\\GLOF\\gradient_boosting"

# Ensure dataset directory exists
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Dataset directory '{image_dir}' not found. Please check the path.")

os.makedirs(mask_dir, exist_ok=True)  # Create mask directory if it doesn't exist

glacier_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpg", ".png"))]

if not glacier_images:
    raise FileNotFoundError(f"❌ No images found in '{image_dir}'. Please check the dataset.")

print(f"Found {len(glacier_images)} glacier images. Generating masks...")

def generate_mask(image):
    """Automatically segments glacier vs. non-glacier using K-Means."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.reshape((-1, 3))  # Flatten image

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Fix FutureWarning for `n_init`
    kmeans.fit(image_flat)
    segmented = kmeans.labels_.reshape(image.shape[:2])  # Reshape to original size
    
    return (segmented * 255).astype(np.uint8)  # Convert to binary mask (0, 255)

# Process each glacier image and generate its mask
for img_path in glacier_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Warning: Failed to load {img_path}. Skipping.")
        continue
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = generate_mask(img)  # Generate mask

    # Save mask image
    mask_filename = os.path.join(mask_dir, os.path.basename(img_path).replace(".jpg", "_mask.png").replace(".png", "_mask.png"))
    cv2.imwrite(mask_filename, mask)
    print(f"✅ Saved mask: {mask_filename}")

print("✅ Glacier masks generated successfully!")

# ---------------------------
# 2️⃣ Load Images & Generated Masks for Training
# ---------------------------

def load_dataset(image_dir, mask_dir):
    """Loads images and corresponding masks as NumPy arrays."""
    images = []
    masks = []
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace(".jpg", "_mask.png").replace(".png", "_mask.png"))

        if not os.path.exists(mask_path):
            print(f"⚠️ Warning: Mask not found for {img_file}. Skipping.")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        if img is None or mask is None:
            print(f"⚠️ Warning: Failed to load {img_file}. Skipping.")
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize images and convert masks to binary (0 or 1)
        img = img / 255.0  # Normalize image pixels
        mask = (mask > 127).astype(np.uint8)  # Convert to binary (0,1)

        images.append(img)
        masks.append(mask)

    if not images or not masks:
        raise RuntimeError("❌ No valid image-mask pairs found. Please check your dataset.")

    return np.array(images), np.array(masks)

# Load dataset
X, Y = load_dataset(image_dir, mask_dir)

# Reshape dataset for Gradient Boosting
X = X.reshape(X.shape[0], -1)  # Flatten images for training
Y = Y.reshape(Y.shape[0], -1)  # Flatten masks for classification

# Split dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print dataset shapes for debugging
print(f"✅ X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"✅ X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

# ---------------------------
# 3️⃣ Train Gradient Boosting for Glacier Segmentation
# ---------------------------

print("🚀 Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, Y_train.ravel())

# Predict on validation set
gb_pred = gb_model.predict(X_val)

# Calculate evaluation metrics
gb_accuracy = accuracy_score(Y_val.ravel(), gb_pred)
gb_iou = jaccard_score(Y_val.ravel(), gb_pred, average='binary')
gb_dice = f1_score(Y_val.ravel(), gb_pred, average='binary')

print(f"✅ Gradient Boosting Accuracy: {gb_accuracy:.4f}")
print(f"✅ Gradient Boosting IoU Score: {gb_iou:.4f}")
print(f"✅ Gradient Boosting Dice Score: {gb_dice:.4f}")
print(classification_report(Y_val.ravel(), gb_pred))
