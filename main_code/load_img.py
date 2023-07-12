from torchvision import datasets, transforms
import torch

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch Tensor data type
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize Tensor images
])

# Load images
image_folder = datasets.ImageFolder('../spectrum_images', transform=transform)

# Create a DataLoader
data_loader = torch.utils.data.DataLoader(image_folder, batch_size=32, shuffle=True)
