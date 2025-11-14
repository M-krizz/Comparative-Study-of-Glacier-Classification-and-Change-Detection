import tensorflow as tf
import tensorflow_hub as hub

# Load DeepLabV3+ from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/deeplabv3/1")

# Test model on a sample image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "D:\\GLOF\\New Code Samples\\Siachen_Glacier_2014.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
image = np.expand_dims(image, axis=0) / 255.0  # Normalize

# Predict segmentation mask
output = model(image)["semantic_logits"]
mask = np.argmax(output, axis=-1)[0]

# Display results
plt.imshow(mask, cmap="gray")
plt.title("DeepLabV3+ Glacier Segmentation")
plt.show()
