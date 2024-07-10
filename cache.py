import os
import shutil
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

workers = 4

projectnb = f'./data/dummy_data'   # projectnb
engnas= f'/ad/eng/research/eng_research_cisl/rsyed/dummy_data' # engnas

# Change these paths accordingly 
data_path = projectnb
cache_path = f'/scratch/rsyed/data'

class DataCache:
    def __init__(self, files: list[str]):
        self.files = files
        # Make data directory in cache in case it doesnt exist
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.in_cache = set(os.listdir(cache_path))
    
    # Return data location in cache if there, otherwise return original path but copy data over to cache too
    def validate_cache(self, fname: str) -> str:
        if fname in self.in_cache:
            return os.path.join(cache_path, fname)
        else:
            self.copy_to_cache(fname)
            return os.path.join(data_path, fname)
    
    # Data copying should be done in background as to not make model wait
    def copy_to_cache(self, fname: str):
        src = os.path.join(data_path, fname)
        dst = os.path.join(cache_path, fname)
        shutil.copy(src, dst)
        self.in_cache.add(fname)

    # Cache will be attempted to be accessed every get_item call in dataset class
    def get_path(self, idx):
        fname = self.files[idx]
        return self.validate_cache(fname)

class CustomImageDataset(Dataset):
    def __init__(self, files: list[str], transform=None):
        self.files = files
        self.transform = transform
        self.cache = DataCache(files)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.cache.get_path(idx)
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image

# Generate list of files
files = os.listdir(data_path)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
dataset = CustomImageDataset(files, transform=transform)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=workers) 

# Model With Customizable Layers
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=1):
        super(SimpleCNN, self).__init__()
        self.layers = num_conv_layers
        
        # Define the convolutional layers dynamically
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=3 if i == 0 else 16, out_channels=16, kernel_size=3, stride=1, padding=1)
            for i in range(num_conv_layers)
        ])
        
        self.fc1 = nn.Linear(16 * 28 * 28, 10)  

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)
        
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return x
    
test_layers = (20,30,40,50,60)
for layers in test_layers:
    model = SimpleCNN(num_conv_layers=layers)                     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    start = time.time()  # Start timer
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.randint(0, 10, (images.size(0),)) # Fake labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}")
    end = time.time()   # End timer
    timer = end-start    # Calculate time

    print(f"Training finished with Layers: {layers} in Time: {timer:.2f} seconds")