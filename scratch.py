import os
import shutil
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

local_data_path = f'./data/dummy_data'   # Local data
ENGNAS_data_path = f'/ad/eng/research/eng_research_cisl/rsyed/dummy_data' # ENGNAS

# Change these paths accordingly 
data_path = ENGNAS_data_path
scratch_path = f'/scratch/rsyed/data'

class ScratchCache:
    def __init__(self, files: list[str]):
        self.files = files
        # Make data directory in scratch in case it doesnt exist
        if not os.path.exists(scratch_path):
            os.makedirs(scratch_path)
        self.in_scratch = set(os.listdir(scratch_path))
    
    # Return data location in scratch if there, otherwise return original path but copy data over to scratch too
    def validate_scratch(self, fname: str) -> str:
        if fname in self.in_scratch:
            return os.path.join(scratch_path, fname)
        else:
            self.copy_to_scratch(fname)
            return os.path.join(data_path, fname)
    
    # Data copying should be done in background as to not make model wait
    def copy_to_scratch(self, fname: str):
        def copy():
            src = os.path.join(data_path, fname)
            dst = os.path.join(scratch_path, fname)
            shutil.copy(src, dst)
            self.in_scratch.add(fname)
        threading.Thread(target=copy).start()

    # Scratch will be attempted to be accessed every get_item call in dataset class
    def get_path(self, idx):
        fname = self.files[idx]
        return self.validate_scratch(fname)

class CustomImageDataset(Dataset):
    def __init__(self, files: list[str], transform=None):
        self.files = files
        self.transform = transform
        self.cache = ScratchCache(files)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.cache.get_path(idx)
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        label = int(os.path.splitext(os.path.basename(file_path))[0].split('_')[1])  # Extract label from file name
        return image, label

# Generate list of files
files = os.listdir(data_path)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
dataset = CustomImageDataset(files, transform=transform)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4) # Play around with num workers

# Example Usage:
# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 10)  # assuming image size is 28x28 and we have 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc1(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Training finished.")
