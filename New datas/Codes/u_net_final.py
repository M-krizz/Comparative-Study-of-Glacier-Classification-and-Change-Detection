import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
                                     Conv2DTranspose, concatenate)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Define the U-Net model
# ---------------------------
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder (Downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder (Upsampling)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model

# -----------------------------------------------------------
# 2. Load images and generate pseudo ground truth masks
#    (No Gaussian blur to keep edges sharp)
# -----------------------------------------------------------
def load_images_and_generate_masks(image_dir, img_size=(256, 256)):
    images, masks = [], []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot open {img_path}")
            continue
        
        img_resized = cv2.resize(img, img_size)
        img_normalized = img_resized / 255.0
        images.append(img_normalized)
        
        # Generate pseudo mask using direct Otsu's thresholding (no blur)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# ---------------------------
# Main script
# ---------------------------
if __name__ == "_main_":
    image_dir = "Images"
    X, Y = load_images_and_generate_masks(image_dir)
    print(f"Images: {X.shape}, Pseudo Masks: {Y.shape}")

    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, 
                                                      test_size=0.2, 
                                                      random_state=42)

    # Build & compile U-Net
    model = unet_model(input_size=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        batch_size=8, epochs=25)

    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save model
    model.save("unet_glacier_model.keras")

    # Prediction function
    def predict_image(model, image_path, img_size=(256, 256)):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open '{image_path}'.")
        
        img_resized = cv2.resize(img, img_size)
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        pred_mask = model.predict(img_input)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        return img_resized, pred_mask

    # Test on a new image
    test_image_path = "Images/Siachen_Glacier_2020.jpg"
    original_img, predicted_mask = predict_image(model, test_image_path)

    # Display results with nearest interpolation
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[:, :, 0], cmap="gray", interpolation='nearest')
    plt.title("Predicted Glacier Mask")
    plt.axis('off')

    plt.show()