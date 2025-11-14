import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Build SegNet Model
def build_segnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preprocess Image
def preprocess_image(image_path, input_shape):
    image = load_img(image_path, target_size=input_shape[:2])
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Decode Segmentation Mask
def decode_segmentation_mask(mask, num_classes):
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    mask = np.argmax(mask, axis=-1)
    color_mask = colors[mask]
    return color_mask

# Compare Images for Changes
def compare_images_for_changes(image_path1, image_path2, year1, year2, model, input_shape, num_classes, glacier_class=0):
    img1 = preprocess_image(image_path1, input_shape)
    img2 = preprocess_image(image_path2, input_shape)
    mask1 = model.predict(img1)[0]
    mask2 = model.predict(img2)[0]
    decoded_mask1 = decode_segmentation_mask(mask1, num_classes)
    decoded_mask2 = decode_segmentation_mask(mask2, num_classes)
    class_mask1 = np.argmax(mask1, axis=-1) == glacier_class
    class_mask2 = np.argmax(mask2, axis=-1) == glacier_class
    change_mask = class_mask1 != class_mask2
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Segmentation {year1}")
    plt.imshow(decoded_mask1)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title(f"Segmentation {year2}")
    plt.imshow(decoded_mask2)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title(f"Changes: {year1} to {year2}")
    plt.imshow(change_mask, cmap='hot')
    plt.axis('off')
    plt.show()
    total_pixels = change_mask.size
    changed_pixels = np.sum(change_mask)
    print(f"Percentage of changed pixels from {year1} to {year2}: {(changed_pixels / total_pixels) * 100:.2f}%")
    return change_mask

# Train SegNet
input_shape = (256, 256, 3)
num_classes = 3
data_gen_args = dict(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen = ImageDataGenerator(**data_gen_args)
train_data = datagen.flow_from_directory('d:\GLOF\Dataset_Images', target_size=input_shape[:2], batch_size=16, class_mode='categorical')
segnet_model = build_segnet(input_shape, num_classes)
segnet_model.fit(train_data, epochs=20)
segnet_model.save('segnet_glacier_model.h5')

# Execute Comparison
image_paths = ['Siachen_Glacier_2014.jpg', 'Siachen_Glacier_2015.jpg', 'Siachen_Glacier_2016.jpg', 'Siachen_Glacier_2017.jpg', 'Siachen_Glacier_2018.jpg', 'Siachen_Glacier_2019.jpg']
years = [2014, 2015, 2016, 2017, 2018, 2019]
for i in range(len(image_paths) - 1):
    compare_images_for_changes(image_paths[i], image_paths[i + 1], years[i], years[i + 1], segnet_model, input_shape, num_classes, glacier_class=0)
