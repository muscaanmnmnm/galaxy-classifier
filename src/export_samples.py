import h5py
import numpy as np
from PIL import Image
import os

# Path to dataset
file_path = "data/archive/Binary_2_5_dataset.h5"
output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

# Open dataset
with h5py.File(file_path, "r") as f:
    X = np.array(f["images"])
    y = np.array(f["labels"])

# Find indices for each class
spiral_indices = np.where(y == 1)[0]
elliptical_indices = np.where(y == 0)[0]

# Pick first 10 from each
spiral_indices = spiral_indices[:10]
elliptical_indices = elliptical_indices[:10]

# Export images
for i, idx in enumerate(spiral_indices):
    img = Image.fromarray((X[idx] * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"spiral_{i}.png"))

for i, idx in enumerate(elliptical_indices):
    img = Image.fromarray((X[idx] * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"elliptical_{i}.png"))

print("✅ Exported 10 spiral and 10 elliptical images to", output_dir)
