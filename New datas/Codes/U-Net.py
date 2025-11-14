import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load Dataset (Ensuring Proper Mask Values)
# ---------------------------

IMAGE_SIZE = 256
image_dir = r"D:\GLOF\Code"
mask_dir = r"D:\GLOF\unet"

# Ensure directories exist
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Error: Directory '{image_dir}' does not exist.")
os.makedirs(mask_dir, exist_ok=True)

# Load images
def load_dataset(image_dir, mask_dir):
    images, masks = [], []

    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", "_mask.png"))

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=-1) / 255.0  # Normalize

        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.expand_dims(mask, axis=-1)  # Ensure shape

        # Convert masks from 255 → 1
        mask = (mask > 127).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

X, Y = load_dataset(image_dir, mask_dir)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ---------------------------
# 2️⃣ Define Improved U-Net Model
# ---------------------------

def build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = Input(shape=input_shape)

    # Encoder with deeper feature extraction
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck with deeper feature extraction
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    c4 = BatchNormalization()(c4)

    # Decoder
    u3 = UpSampling2D((2, 2))(c4)
    u3 = Conv2D(256, (3, 3), activation="relu", padding="same")(u3)
    u3 = Concatenate()([c3, u3])

    u2 = UpSampling2D((2, 2))(u3)
    u2 = Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
    u2 = Concatenate()([c2, u2])

    u1 = UpSampling2D((2, 2))(u2)
    u1 = Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    u1 = Concatenate()([c1, u1])

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u1)

    return Model(inputs, outputs)

# ---------------------------
# 3️⃣ Train Optimized U-Net Model
# ---------------------------

unet_model = build_unet()
unet_model.compile(optimizer=Adam(learning_rate=0.0005), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
model_checkpoint = ModelCheckpoint("unet_best_model.h5", save_best_only=True, monitor="val_loss")

history = unet_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=25,
    batch_size=8,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Save final model
unet_model.save("unet_final_model.h5")
print("Model trained and saved.")

# ---------------------------
# 4️⃣ Evaluate Model Performance
# ---------------------------

Y_pred = unet_model.predict(X_val)
Y_pred = (Y_pred > 0.5).astype(np.uint8)

# Flatten for evaluation
Y_val_flat = Y_val.flatten()
Y_pred_flat = Y_pred.flatten()

# Calculate metrics
unet_accuracy = accuracy_score(Y_val_flat, Y_pred_flat)
unet_iou = jaccard_score(Y_val_flat, Y_pred_flat, average="binary")
unet_f1 = f1_score(Y_val_flat, Y_pred_flat, average="binary")
unet_precision = precision_score(Y_val_flat, Y_pred_flat, average="binary")
unet_recall = recall_score(Y_val_flat, Y_pred_flat, average="binary")

print(f"U-Net Accuracy: {unet_accuracy:.4f}")
print(f"U-Net IoU Score: {unet_iou:.4f}")
print(f"U-Net Precision: {unet_precision:.4f}")
print(f"U-Net Recall: {unet_recall:.4f}")
print(f"U-Net F1 Score: {unet_f1:.4f}")