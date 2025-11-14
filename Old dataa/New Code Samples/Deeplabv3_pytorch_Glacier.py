import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Load a pre-trained DeepLabV3 model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((520, 520)),  # Resize to match DeepLabV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an input image
image_path = "D:\\GLOF\\New Code Samples\\Siachen_Glacier_2019.jpg"  # Change this to your image path
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)["out"]
    
# Get predicted segmentation mask
output_predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# Display original image and segmentation result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(output_predictions, cmap="jet")
ax[1].set_title("Segmentation Mask")
ax[1].axis("off")

plt.show()
