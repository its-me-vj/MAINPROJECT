import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import random

# ✅ Define CSRNet Model
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.ReLU()  # Ensures non-negative predictions
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# ✅ Load Model
def load_model(weights_path):
    model = CSRNet()
    model_weights = torch.load(weights_path, map_location=torch.device('cpu'))

    # ✅ Filter out incorrect weight keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_weights.items() if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    return model

# ✅ Paths
model_path = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\crowd_counting_model\weights.pth"
image_dir = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\test_data\images"

# ✅ Load Model
model = load_model(model_path)

# ✅ Preprocessing Transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Ensure same as training
])

# ✅ Select 5 Random Images from Dataset
image_files = random.sample(os.listdir(image_dir), 5)
image_paths = [os.path.join(image_dir, img) for img in image_files]

# ✅ Process and Display Images One by One
for image_path in image_paths:
    # Load Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process Image
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(input_tensor)

    # Convert Output to Density Map
    density_map = output.squeeze(0).squeeze(0).cpu().numpy()
    predicted_count = int(density_map.sum())  # Summing the density map gives the count

    # ✅ Display Image & Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original Image
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    # Density Map
    axes[1].imshow(density_map, cmap='jet', interpolation='nearest')
    axes[1].axis("off")
    axes[1].set_title(f"Predicted Count: {predicted_count}")

    plt.tight_layout()
    plt.show()  # Waits for user to close before showing the next image
