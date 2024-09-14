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
### Initialization
Here is a relevant snippet from the `__init__` function of the dataset class for the model:
```
self.cache = DataCache(self.data_path, self.cache_path)
```
The way it works is simple. Upon initialization, the original data location (data path) and training data location (cache path) are required, so they should also be input arguments for the overlying dataset class. The `files` list is simply just a list of all the items in the orignial data. The `in_cache` set is created in order to keep track of the current items within the cache path. 

### Usage
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
Near the bottom of `cache.py` many lines of code to run and plot the training time results of using this cache in different scenarios can be seen. The results will be explained in detail here. Note that the raw data can be seen in the `results/` directory in this repository in the form of .csv files.
![image](https://github.com/user-attachments/assets/4337ca1d-e7e2-493f-9610-d501882df578)
Above is a logarithmic graph showing the training times in seconds with a data size of 500 33x224x224 images and ground truth of 500 24x224x224 images. 

### Without the Data Cache
`Projectnb` in the SCC is the 'local' data storage location. When the data is stored in this local location, the training time seems to be about average as seen by the green line. `ENGNAS` is the 'external' data storage location in this case. When the data is stored in ENGNAS, the training time is obviously much slower, as seen by the purple line. The multithreading counterparts of both of these scenarios (num_workers=4) are obviously significantly faster as seen by the red and brown lines respectively. 

### With the Data Cache
When the cache was integrated into the model, the training times for the local location was slightly faster even though not significantly, as seen by the blue line. On the other hand, the external storage saw a massive improvement in training times, as seen by the orange line. The performance of training with data at an external location with the use of this cache_dataloading feature allowed for the training times to be very comparable to that of training with data from a local location. This is a significant improvement and very important find. 

When the same tests were performed with 5000 image datasets of the same size instead, the results were consistent, as seen here:
![image](https://github.com/user-attachments/assets/1cd93fb6-2223-4ff1-825b-40bbc34a1b52)
This logarithmic graph shows that the relative difference between all the different events' training times are about the same.

## Conclusion
From these results, it is safe to conclude that using the DataCache class is worth it in almost all scenarios. For local data, a slight improvement can be seen, while for external data, significant improvement can be seen. Multithreading is still clearly more efficient, but scratch cache overcomes some of its downsides such as having a lower priority job, using high amounts of memory/large overhead, and potentially causing CPU overusage (leading to job being reaped/killed). 

However, there is possibility for a solution even better than this. Currently, this cache_dataloading feature cannot be used with multiple workers (multithreading) since there is potential for corruption, as the model might access data not yet done copying or the same item might end up being copied multiple times. By making the DataCache class thread safe, this downside can be overcome, and multithreading can be combined with this project in order to act as the most optimal solution for data loading. Based on the results from the graph, it is clear that multiple workers with the cache will cause the massive time saves of both methods to combine, leading to the best case training times. 

Hopefully, this class can be seamlessly integrated into your models and help with optimizing training. If you have any further questions or spot any errors, please contact at me at rsyed@bu.edu.
