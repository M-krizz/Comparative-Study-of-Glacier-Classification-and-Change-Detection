import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load Glacier Images & Generate Masks Automatically
# ---------------------------

IMAGE_SIZE = 256
image_dir = "D:\\GLOF\\Code"  
mask_dir = "D:\\GLOF\\unet"

os.makedirs(mask_dir, exist_ok=True)  # Create mask directory if not exists

glacier_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpg", ".png"))]

print(f"Found {len(glacier_images)} glacier images. Generating masks...")

def generate_mask(image):
    """Automatically segments glacier vs. non-glacier using K-Means."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.reshape((-1, 3))  # Flatten image

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(image_flat)  # 2 Clusters (Glacier, Non-Glacier)
    segmented = kmeans.labels_.reshape(image.shape[:2])  # Reshape to original size
    
    return (segmented * 255).astype(np.uint8)  # Convert to binary mask (0, 255)

# Process each glacier image and generate its mask
for img_path in glacier_images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        continue
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = generate_mask(img)  # Generate mask

    # Save mask image
    mask_filename = os.path.join(mask_dir, os.path.basename(img_path).replace(".jpg", "_mask.png"))
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
        mask_path = os.path.join(mask_dir, file.replace(".jpg", "_mask.png"))

        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=-1) / 255.0  # Normalize & add channel

        if not os.path.exists(mask_path):
            print(f"❌ Mask not found: {mask_path}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"❌ Failed to read mask: {mask_path}")
            continue

        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.expand_dims(mask, axis=-1) / 255.0  # Normalize & add channel

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load dataset
X, Y = load_dataset(image_dir, mask_dir)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ---------------------------
# 3️⃣ Define U-Net Model
# ---------------------------

def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = Conv2D(128, (3, 3), activation="relu", padding="same")(u1)
    u1 = Concatenate()([c2, u1])

    u2 = UpSampling2D((2, 2))(u1)
    u2 = Conv2D(64, (3, 3), activation="relu", padding="same")(u2)
    u2 = Concatenate()([c1, u2])

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u2)
    
    return Model(inputs, outputs)

# Build & Compile U-Net Model
unet_model = build_unet()
unet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
unet_model.summary()

# ---------------------------
# 4️⃣ Train the U-Net Model
# ---------------------------

history = unet_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=8
)

# Save trained model
unet_model.save("unet_glacier_model.h5")
print("✅ U-Net model trained and saved!")