import os
import shutil
import threading
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Change these paths accordingly 
data_path = f'./data/dummy_data'   # need to test with ENGNAS
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
        print(idx)
        file_path = self.cache.get_path(idx)
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        label = int(os.path.splitext(os.path.basename(file_path))[0].split('_')[1])  # Extract label from file name
        return image, label

# Generate list of files
files = os.listdir(data_path)
transform = transforms.ToTensor()
dataset = CustomImageDataset(files, transform=transform)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4) # Play around with num workers

# Example usage
for images, labels in train_loader:
    print(images.size(), labels)
