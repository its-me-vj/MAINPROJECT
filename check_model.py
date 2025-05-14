import torch

# Load the weights.pth file
weights_path = "C:/Users/HP/Desktop/Crowd Density/Crowd Density/crowd_counting_model/weights.pth"
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# Print the state dict to inspect the layers and their shapes
for name, param in state_dict.items():
    print(f"Layer name: {name}, Shape: {param.shape}")
