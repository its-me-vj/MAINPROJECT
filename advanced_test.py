import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from torchvision import models
from models.csrnet import CSRNet  # Ensure correct path to CSRNet
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VGG-16 (feature extractor)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
features = vgg16.features  # Extract convolutional layers only (remove classifier)

# Define CSRNet-based Model
class CSRNet(nn.Module):
    def __init__(self, features):
        super(CSRNet, self).__init__()
        self.frontend = features  # Use VGG-16 features as frontend
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1)  # Output single-channel density map
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# Load the trained model
model = CSRNet(features).to(device)  # Ensure CSRNet is correctly initialized with VGG features

# Load the checkpoint
checkpoint_path = r"C:\Users\hp\Desktop\Crowd Density\crowd_counting_model\tuned.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Get the model state dict
model_dict = model.state_dict()

# Load the state_dict from the checkpoint
pretrained_dict = checkpoint

# Only load weights for layers that exist in both the model and the checkpoint
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# Update the model's state dict with the compatible weights
model_dict.update(pretrained_dict)

# Load the state_dict into the model
model.load_state_dict(model_dict)

# Switch model to evaluation mode
model.eval()

# Define test image directories
TEST_DIRS = [
    r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\test_data\images",
    r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_B_final\test_data\images"
]

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 400)),  # Resize to training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load {image_path}")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_tensor = transform(img).unsqueeze(0).to(device)

    return img, img_tensor

# Predict density map
def predict_density(image_path):
    img, img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None

    with torch.no_grad():
        output = model(img_tensor)  # Forward pass

    # Ensure the output is resized to match the input image
    output_resized = F.interpolate(output, size=img.shape[:2], mode='bilinear', align_corners=False)

    density_map = output_resized.squeeze(0).squeeze(0).cpu().numpy()
    density_map = np.maximum(density_map, 0)  # Ensure non-negative values

    # Calculate crowd count by summing the values of the density map
    crowd_count = density_map.sum()

    return img, density_map, crowd_count

# Get first 5 images from each test dataset
test_images = []
for test_dir in TEST_DIRS:
    if os.path.exists(test_dir):
        images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".png"))]
        test_images.extend(images[:5])  # Take first 5 images

# Plot results in a graph
num_images = len(test_images)
plt.figure(figsize=(10, num_images * 2))

# Plot results in separate figures for each image
for i, img_path in enumerate(test_images):
    img, density_map, crowd_count = predict_density(img_path)
    if img is None or density_map is None:
        continue  # Skip images that failed to load

    # Create a new figure for each image and its density map
    plt.figure(figsize=(10, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image {i + 1}")
    plt.axis("off")

    # Plot the density map
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap="jet")
    plt.title(f"Density Map {i + 1}\nCrowd Count: {int(crowd_count)}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()  # Show each image and density map in a separate window
