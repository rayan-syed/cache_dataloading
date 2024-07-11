import os
import shutil
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import matplotlib.pyplot as plt    

class DataCache:
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path
        self.files = os.listdir(self.data_path)

        # Make data directory in cache in case it doesnt exist
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        # Set of cache contents
        self.in_cache = set(os.listdir(self.cache_path))
    
    # Copy data to cache and then return cache path
    def validate_cache(self, fname: str) -> str:
        if fname not in self.in_cache:
            self.copy_to_cache(fname)
        return os.path.join(self.cache_path, fname)
    
    # Data copying should be done in background as to not make model wait
    def copy_to_cache(self, fname: str):
        src = os.path.join(self.data_path, fname)
        dst = os.path.join(self.cache_path, fname)
        shutil.copy(src, dst)
        self.in_cache.add(fname)

    # Cache will be attempted to be accessed every get_item call in dataset class
    def get_path(self, idx):
        fname = self.files[idx]
        return self.validate_cache(fname)

class CustomImageDataset(Dataset):
    def __init__(self, data_path, cache_path, transform=None, use_cache=True):
        self.transform = transform
        self.data_path = data_path
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.files = os.listdir(self.data_path)
        if self.use_cache:
            self.cache = DataCache(self.data_path, self.cache_path)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.use_cache:
            file_path = self.cache.get_path(idx)
        else:
            file_path = os.path.join(self.data_path, self.files[idx])
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image

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
    
# Function to train the model and measure time elapsed
def train_model(num_layers, data_path, cache_path, use_cache, num_workers):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(data_path=data_path, cache_path=cache_path, transform=transform, use_cache=use_cache)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    
    model = SimpleCNN(num_conv_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    start = time.time()     # Start timer
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images in train_loader:
            optimizer.zero_grad()
            outputs = model(images) 
            labels = torch.randint(0, 10, (images.size(0),))     # Fake Labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    end = time.time()       # End timer
    return end - start

# Relevant paths
projectnb = f'/projectnb/tianlabdl/rsyed/cache_dataloading/data/dummy_data'  
engnas = f'/ad/eng/research/eng_research_cisl/rsyed/dummy_data' 
scratch = f'/scratch/rsyed/data'

# Configurations to test
configurations = [
    (True, projectnb, 1, "Projectnb w cache"),
    (True, engnas, 1, "ENGNAS w cache"),          
    (False, projectnb, 1, "Projectnb w/o cache & 1 worker"),   
    (False, projectnb, 4, "Projectnb w/o cache & 4 workers"),   
    (False, engnas, 1, "ENGNAS w/o cache & 1 worker"),         
    (False, engnas, 4, "ENGNAS w/o cache & 4 workers")          
]

# Collect timing data
results = {config: [] for config in configurations}
test_layers = [20, 30, 40, 50, 60]
for layers in test_layers:
    print(f"on layer: {layers}")
    for config in configurations:
        use_cache, data_path, workers, _temp_ = config
        time_taken = train_model(num_layers=layers, data_path=data_path, cache_path=scratch, use_cache=use_cache, num_workers=workers)
        results[config].append((layers, time_taken))

# Plot results
plt.figure(figsize=(12, 8))
for config, res in results.items():
    layers, time_taken = zip(*res)  # break up tuple into x,y
    label = config[3]
    plt.plot(layers, time_taken, label=label)

plt.xlabel('Number of Layers')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Number of Layers')
plt.legend()
plt.savefig('cache_comparison.png')
