import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from models.csrnet import CSRNet
from data.dataset import CrowdDataset

# Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_paths = [
    os.path.join(BASE_DIR, "dataset", "part_A_final", "train_data"),
    os.path.join(BASE_DIR, "dataset", "part_B_final", "train_data")
]

# Verify dataset directories
train_datasets = []
for path in dataset_paths:
    img_dir = os.path.join(path, "images")
    gt_dir = os.path.join(path, "ground_truth")

    if os.path.exists(img_dir) and os.path.exists(gt_dir):
        transform = transforms.Compose([transforms.ToTensor()])
        train_datasets.append(CrowdDataset(img_dir, gt_dir, transform))
    else:
        print(f"Warning: Missing dataset at {path}")

# Ensure at least one dataset exists
if not train_datasets:
    raise FileNotFoundError("No valid training dataset found!")

# Combine datasets
train_dataset = ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
def train(model, train_loader, epochs):
    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for images, density_maps in train_loader:
            images, density_maps = images.to(device), density_maps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, density_maps)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/50, Loss: {epoch_loss / len(train_loader)}")

if __name__ == "__main__":
    train(model, train_loader, 100)
    model_save_path = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\crowd_counting_model\weights.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model trained and saved successfully at {model_save_path}!")
