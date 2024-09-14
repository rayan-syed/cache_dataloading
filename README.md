# Python Class to Optimize Dataloading Efficiency
This repository aims to optimize dataloading efficiency during model training, especially in the case of data being stored in faraway hard drives. This repository was tested on [Boston University's Shared Computing Cluster (SCC)](https://www.bu.edu/tech/support/research/computing-resources/scc/), the batch system of which is based on the [Sun Grid Engine](https://gridscheduler.sourceforge.net/) (SGE) scheduler.

## Idea and Motivation
Training models with local data tends to be seemless, but when the data is stored on an external hard drive located farther away in the network, training tends to take much longer. This could be overcome by the existence of a local 'cache' to copy all of the data locally during the first epoch of model training. For all susbequent epochs, the data would be accessed locally rather than externally, significantly reducing I/O latency and reducing the overall training time of the model. 

In the case of the SCC, every GPU node has its own 'scratch directory,' allowing for temporary storage while the GPU is in use. Here is a snippet from SCC's Documentation:
```
There are occasions in which the use of a local scratch disk storage may be beneficial: like gaining access to additional (temporary) storage as well as to take advantage of its close proximity to the compute node for more efficient I/O operations.
```

This implementation for the cache_dataloading idea is created on and tested with the BU's SCC's scratch directory, but the code is developed in such a way that it can be easily integrated into any model. Details on the code will be described later.

## The DataCache Class
Here is the most relevant snippet of code from `cache.py` to be integrated into any model:
```
class DataCache:
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path
        self.files = os.listdir(self.data_path)
        self.in_cache = set()

        # Make data directory in cache in case it doesnt exist
        os.makedirs(self.cache_path, exist_ok=True)
    
    # Copy data to cache if not already in and then return cache path
    def validate_cache(self, fname: str):
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
```
# Initialization.
Here is a relevant snippet from the `__init__` function of the dataset class for the model:
```
self.cache = DataCache(self.data_path, self.cache_path)
```
The way it works is simple. Upon initialization, the original data location (data path) and training data location (cache path) are required, so they should also be input arguments for the overlying dataset class. The `files` list is simply just a list of all the items in the orignial data. The `in_cache` set is created in order to keep track of the current items within the cache path. 

# Usage
Here a relevant snippet from the `__getitem__` function of the dataset class for the model:
```
file_path_main = self.cache.get_path(idx)
```
This `get_path` function will utilize helper functions (as seen in code snippet above) to do the following things:
1. Check if the item trying to be accessed is already in the cache directory by checking the contents of the `in_cache` set
2. If the item is not in the cache directory, the item will be copied over from the source path and the item will be added to the `in_cache` set
3. The path of the cache directory will be returned, so it can be used by the dataset class

For all epochs after the first, the cache directory will already be loaded with all relevant data and the `in_cache` set will reflect this. However, please note that the `in_cache` set may reset if one does not specify `persistent_workers=True` when intializing the dataloader, as seen here:
```
train_loader = DataLoader(dataset, batch_size=128, prefetch_factor=4, shuffle=True, num_workers=num_workers, persistent_workers=True)
```
The lack of not specifying this can lead to the unintended behavior of every item being copied every epoch, which is more inefficient than the model before implementing this cache_dataloading feature.

## Results
Near the bottom of `cache.py` many lines of code to run and plot the training time results of using this cache in different scenarios can be seen. The results will be explained in detail here.
![image](https://github.com/user-attachments/assets/d600e945-361c-45a2-ae42-1413b86eefe4)
