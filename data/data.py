import os
import numpy as np
from PIL import Image

data_path = './dummy_data'
os.makedirs(data_path, exist_ok=True)

# Generate 5000 random PNGS
for i in range(5000):
    # Generate random pixel values for each channel (RGB)
    random_pixels = np.random.randint(0,256, (256, 256, 3), dtype=np.uint8)
    # Create an image from the random pixel values
    image = Image.fromarray(random_pixels, mode='RGB')
    # Save new image
    image.save(os.path.join(data_path, f'image_{i}.png'))

print("Dummy data generation complete")

