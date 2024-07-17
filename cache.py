import os
import shutil
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tifffile as tiff
import time
import datetime
import matplotlib.pyplot as plt  
import csv  

class DataCache:
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path
        self.files = os.listdir(self.data_path)

        # Make data directory in cache in case it doesnt exist
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
    
    # Copy data to cache if not already in and then return cache path
    def validate_cache(self, fname: str):
        if fname not in set(os.listdir(self.cache_path)):
            self.copy_to_cache(fname)
        return os.path.join(self.cache_path, fname)
    
    # Data copying should be done in background as to not make model wait
    def copy_to_cache(self, fname: str):
        src = os.path.join(self.data_path, fname)
        dst = os.path.join(self.cache_path, fname)
        shutil.copy(src, dst)

    # Cache will be attempted to be accessed every get_item call in dataset class
    def get_path(self, idx):
        fname = self.files[idx]
        return self.validate_cache(fname)

class CustomImageDataset(Dataset):
    def __init__(self, data_path, cache_path, use_cache, ground_truth=None, transform=None):
        self.transform = transform
        self.ground_truth = ground_truth
        self.data_path = data_path
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.files = os.listdir(self.data_path)
        if self.use_cache:
            self.cache = DataCache(self.data_path, self.cache_path)
            if self.ground_truth:
                self.ground_truth_cache = DataCache(self.ground_truth, self.cache_path)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.use_cache:
            file_path_main = self.cache.get_path(idx)
            if self.ground_truth:
                temp = self.ground_truth_cache.get_path(idx)
        else:
            file_path_main = os.path.join(self.data_path, self.files[idx])
            if self.ground_truth:
                temp = os.path.join(self.ground_truth, self.files[idx])
        
        # Use tifffile to read the TIFF image
        image = tiff.imread(file_path_main)
        if self.ground_truth:
            temp = tiff.imread(temp)        # simulate reading ground truth but dont do anything with it
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)
        
        return image

# Model With Customizable Layers
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=1, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.layers = num_conv_layers
        
        # Define the convolutional layers dynamically
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels if i == 0 else 16, out_channels=16, kernel_size=3, stride=1, padding=1)
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
    
# Function to clear the cache directory
def clear_cache(cache_path):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

# Function to train the model and measure time elapsed
def train_model(num_layers, input_channels, data_path, cache_path, use_cache, num_workers, ground_truth_path=None):
    transform = transforms.Compose([
        transforms.Resize((28, 28))
    ])
    dataset = CustomImageDataset(data_path=data_path, cache_path=cache_path, use_cache=use_cache, ground_truth=ground_truth_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    
    model = SimpleCNN(num_conv_layers=num_layers, input_channels=input_channels)
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
dataset = '33x224x224'
ground_truth = '24x224x224'
projectnb = f'/projectnb/tianlabdl/rsyed/cache_dataloading/data'  
engnas = f'/ad/eng/research/eng_research_cisl/rsyed' 
scratch = '/scratch/rsyed/data'

# Configurations to test
configurations = [
    (True, projectnb, 1, "Projectnb w cache & 1 worker"),
    (True, engnas, 1, "ENGNAS w cache & 1 worker"), 
    (False, projectnb, 1, "Projectnb w/o cache & 1 worker"),   
    (False, projectnb, 4, "Projectnb w/o cache & 4 workers"),   
    (False, engnas, 1, "ENGNAS w/o cache & 1 worker"),         
    (False, engnas, 4, "ENGNAS w/o cache & 4 workers")          
]

# Collect timing data
print("\nStarting timing...\n")
in_channels = 33            # Specify input channels for model
results = {config: [] for config in configurations}
test_layers = [20, 30, 40, 50, 60]
for layers in test_layers:
    for config in configurations:
        use_cache, path, workers, name = config
        data_path = f'{path}/{dataset}'
        ground_truth_path = f'{path}/{ground_truth}'
        print("Clearing cache to simulate new run...")
        clear_cache(scratch)        # Clear the cache directory for accurate testing
        print(f"Starting to train {layers} layer model with config: {name} @ {datetime.datetime.now()}...")
        # Remove ground_truth_path variable if timing w/o ground truth
        time_taken = train_model(num_layers=layers, input_channels=in_channels, data_path=data_path, cache_path=scratch, use_cache=use_cache, num_workers=workers, ground_truth_path=ground_truth_path)
        print(f"Training took: {time_taken}\n")
        results[config].append((layers, time_taken))
print("Results completed.\n")

res_name = "ground_truth"

# Make results directory
if not os.path.exists(f'results/{res_name}'):
    os.makedirs(f'results/{res_name}')

# Save results to a CSV file
print("Saving to csv file...")
csv_filename = f'results/{res_name}/{res_name}.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Configuration', 'Number of Layers', 'Time (seconds)'])
    for config, res in results.items():
        for layers, time_taken in res:
            writer.writerow([config[3], layers, time_taken])

# Plot results
print("Plotting results...")
plt.figure(figsize=(12, 8))

# Linear
for config, res in results.items():
    layers, time_taken = zip(*res)  # Break up tuple into x,y
    label = config[3]
    plt.plot(layers, time_taken, label=label)

plt.xlabel('Number of Layers')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Number of Layers')
plt.legend()
plt.savefig(f'results/{res_name}/{res_name}_linear.png')
print("Linear graph plotted.")
plt.clf()        # Clear the current figure

# Logarithmic
plt.figure(figsize=(12, 8))
for config, res in results.items():
    layers, time_taken = zip(*res)  # Break up tuple into x,y
    label = config[3]
    plt.plot(layers, time_taken, label=label)

plt.yscale('log')
plt.xlabel('Number of Layers')
plt.ylabel('Time (seconds) [log scale]')
plt.title('Training Time vs Number of Layers (Log Scale)')
plt.legend()
plt.savefig(f'results/{res_name}/{res_name}_log.png')
print("Logarithmic graph plotted.")