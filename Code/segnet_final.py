import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load Glacier Images from Directory
# ---------------------------

IMAGE_SIZE = 256
NUM_CLASSES = 2
image_dir = "D:\\GLOF\\Code"  
mask_dir = "D:\\GLOF\\segnet"
os.makedirs(mask_dir, exist_ok=True)

glacier_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg") or img.endswith(".png")]

print(f"Found {len(glacier_images)} glacier images.")

# ---------------------------
# 2️⃣ Generate Glacier Masks Using Segmentation Methods
# ---------------------------

def segment_using_kmeans(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_flat)
    segmented = kmeans.labels_.reshape(image.shape[:2])
    
    return segmented

def segment_using_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# Load SAM model
sam_checkpoint = "sam_vit_h.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

def segment_using_sam(image):
    predictor.set_image(image)
    input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])  # Central point
    input_label = np.array([1])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    return masks[0].astype(np.uint8) * 255

# Process all images and generate masks
for img_path in glacier_images:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Apply different segmentation techniques
    kmeans_mask = segment_using_kmeans(img)
    otsu_mask = segment_using_otsu(img)
    sam_mask = segment_using_sam(img)

    # Combine masks (majority voting)
    final_mask = ((kmeans_mask + otsu_mask + sam_mask) > 255).astype(np.uint8) * 255

    # Save the mask
    mask_filename = os.path.join(mask_dir, os.path.basename(img_path).replace(".jpg", "_mask.png"))
    cv2.imwrite(mask_filename, final_mask)
    print(f"Saved mask: {mask_filename}")

print("✅ Glacier masks generated successfully!")

# ---------------------------
# 3️⃣ Load Masks & Train SegNet Model
# ---------------------------

def load_images(image_dir, mask_dir):
    images, masks = [], []

    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", "_mask.png"))  # Ensure mask name matches

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.expand_dims(mask, axis=-1)
        mask = to_categorical(mask, num_classes=NUM_CLASSES)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load dataset
X, Y = load_images(image_dir, mask_dir)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training on {X_train.shape[0]} images, Validating on {X_val.shape[0]} images")

# ---------------------------
# 4️⃣ Define SegNet Model
# ---------------------------

def build_segnet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(x)
    
    return Model(inputs, outputs)

# Train SegNet
segnet_model = build_segnet()
segnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = segnet_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=8
)

# Save trained model
segnet_model.save("segnet_glacier_model.h5")
print("✅ SegNet model trained and saved!")

# ---------------------------
# 5️⃣ Predict on New Glacier Image
# ---------------------------

test_image_path = "dataset/glacier_images/glacier_2021.jpg"
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
test_image = np.expand_dims(test_image, axis=0)

pred_mask = segnet_model.predict(test_image)
pred_mask = np.argmax(pred_mask[0], axis=-1)

# Display results
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Glacier Segmentation (2021)")
plt.show()