import torch
import numpy as np
import scipy.io
import os
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_size=(256, 256), density_size=(64, 64)):  # Fixed to 64x64
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.image_files = sorted(os.listdir(img_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        self.img_size = img_size
        self.density_size = density_size

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        gt_filename = "GT_" + os.path.basename(img_path).replace(".jpg", ".mat")
        gt_path = os.path.join(self.gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"❌ Missing ground truth file: {gt_path}")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        image = self.transform(image)

        # Load ground truth data
        gt_data = scipy.io.loadmat(gt_path)['image_info'][0, 0][0, 0][0]

        # Initialize density map with resized dimensions
        density_map = np.zeros(self.img_size, dtype=np.float32)

        # Scale points to match resized image
        scale_x = self.img_size[0] / orig_width
        scale_y = self.img_size[1] / orig_height

        for point in gt_data:
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)

            # Ensure x and y are within valid bounds
            x = min(max(x, 0), self.img_size[0] - 1)
            y = min(max(y, 0), self.img_size[1] - 1)

            density_map[y, x] = 1.0

        # Resize density map to match CSRNet output (64x64) using OpenCV
        density_map = cv2.resize(density_map, self.density_size, interpolation=cv2.INTER_CUBIC)

        # Normalize density map so the sum remains the same after resizing
        density_map *= (np.sum(density_map) / (np.sum(density_map) + 1e-8))

        return image, torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.image_files)

# Define dataset paths
IMG_DIR_A = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\train_data\images"
GT_DIR_A = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_A_final\train_data\ground_truth"

IMG_DIR_B = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_B_final\train_data\images"
GT_DIR_B = r"C:\Users\hp\Desktop\Crowd Density\Crowd Density\dataset\part_B_final\train_data\ground_truth"

# Load datasets
dataset_A = CrowdDataset(IMG_DIR_A, GT_DIR_A)
dataset_B = CrowdDataset(IMG_DIR_B, GT_DIR_B)

# Combine both datasets
combined_dataset = ConcatDataset([dataset_A, dataset_B])

dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)

print(f"✅ Combined dataset loaded with {len(combined_dataset)} images.")
