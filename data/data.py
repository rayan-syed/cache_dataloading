import os
import numpy as np
from tifffile import imsave

size = 224
channels = 33
images = 500
name = 'image'

data_path = f'./{images},{channels}x{size}x{size}'
os.makedirs(data_path, exist_ok=True)

# Generate 5000 random TIFF images
for i in range(images):
    # Generate random pixel values for each channel
    random_pixels = np.random.randint(0, 256, (channels, size, size), dtype=np.uint8)
    # Save new image as TIFF
    imsave(os.path.join(data_path, f'{name}_{i}.tiff'), random_pixels)

print("Data generation complete")

