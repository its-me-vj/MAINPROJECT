import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from data.dataset import CrowdDataset  # Import the dataset class
import os

# Define dataset paths
IMG_DIR_A = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\train_data\images"
GT_DIR_A = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\train_data\ground_truth"

IMG_DIR_B = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_B_final\train_data\images"
GT_DIR_B = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_B_final\train_data\ground_truth"

# Check if dataset directories exist
if not all(os.path.exists(path) for path in [IMG_DIR_A, GT_DIR_A, IMG_DIR_B, GT_DIR_B]):
    raise FileNotFoundError("❌ One or more dataset paths are incorrect!")

# Load datasets
dataset_A = CrowdDataset(IMG_DIR_A, GT_DIR_A)
dataset_B = CrowdDataset(IMG_DIR_B, GT_DIR_B)

# Combine both datasets
combined_dataset = ConcatDataset([dataset_A, dataset_B])
train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)

print(f"✅ Combined dataset loaded with {len(combined_dataset)} images.")

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

# Initialize CSRNet model
model = CSRNet(features)

print("✅ CSRNet-based model initialized.")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, density_maps in train_loader:
        images, density_maps = images.to(device), density_maps.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Output shape: [batch, 1, H, W]

        # Ensure density_maps has the correct shape (batch, 1, H, W)
        if density_maps.dim() == 3:
            density_maps = density_maps.unsqueeze(1)  # Convert [batch, H, W] to [batch, 1, H, W]

        # Resize density_maps to match outputs' spatial dimensions
        density_maps = F.interpolate(density_maps, size=outputs.shape[2:], mode="bilinear", align_corners=False)

        print(f"🔍 Model output shape: {outputs.shape}")
        print(f"🔍 Ground truth shape: {density_maps.shape}")

        loss = criterion(outputs, density_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Save the tuned model
os.makedirs(r"C:\Users\hp\Desktop\Crowd Density\crowd_counting_model", exist_ok=True)
save_path = r"C:\Users\hp\Desktop\Crowd Density\crowd_counting_model\tuned.pth"
torch.save(model.state_dict(), save_path)

print(f"✅ Tuned model saved at {save_path}")
print("✅ Tuning script executed successfully.")
