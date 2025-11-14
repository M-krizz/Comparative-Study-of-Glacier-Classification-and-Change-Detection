import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load Glacier Images & Generate Masks Automatically
# ---------------------------

IMAGE_SIZE = 256
image_dir = "D:\GLOF\Code"
mask_dir = "D:\GLOF\SVM"

os.makedirs(mask_dir, exist_ok=True)  # Create mask directory if not exists

glacier_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpg", ".png"))]

print(f"Found {len(glacier_images)} glacier images. Generating masks...")

def generate_mask(image):
    """Automatically segments glacier vs. non-glacier using K-Means."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.reshape((-1, 3))  # Flatten image

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(image_flat)  # Ensure n_init is specified
    segmented = kmeans.labels_.reshape(image.shape[:2])  # Reshape to original size
    
    return (segmented * 255).astype(np.uint8)  # Convert to binary mask (0, 255)

# Process each glacier image and generate its mask
for img_path in glacier_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Warning: Unable to read {img_path}, skipping...")
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
    images, masks = [], []
    
    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", "_mask.png").replace(".png", "_mask.png"))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"⚠️ Warning: Missing image or mask for {file}, skipping...")
            continue
        
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.reshape(-1)  # Flatten image
        
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = mask.reshape(-1)  # Flatten mask
        
        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

X, Y = load_dataset(image_dir, mask_dir)

if len(X) == 0 or len(Y) == 0:
    raise RuntimeError("Error: No valid images or masks found.")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ---------------------------
# 3️⃣ Train Random Forest for Glacier Segmentation
# ---------------------------

print("🚀 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_model.fit(X_train, Y_train.ravel())

# Predict on validation set
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(Y_val.ravel(), rf_pred)
rf_iou = jaccard_score(Y_val.ravel(), rf_pred, average='binary')
rf_dice = f1_score(Y_val.ravel(), rf_pred, average='binary')

print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"✅ Random Forest IoU Score: {rf_iou:.4f}")
print(f"✅ Random Forest Dice Score: {rf_dice:.4f}")
print(classification_report(Y_val.ravel(), rf_pred))
