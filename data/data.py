import os
import numpy as np
from tifffile import imsave

size = 224
channels = 33

data_path = f'./{channels}x{size}x{size}'
os.makedirs(data_path, exist_ok=True)

# Generate 5000 random TIFF images
for i in range(5000):
    # Generate random pixel values for each channel
    random_pixels = np.random.randint(0, 256, (channels, size, size), dtype=np.uint8)
    # Save new image as TIFF
    imsave(os.path.join(data_path, f'image_{i}.tiff'), random_pixels)

print("Data generation complete")

