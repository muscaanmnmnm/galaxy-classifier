import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def load_binary_galaxy_dataset(filepath="data/archive/Binary_2_5_dataset.h5", test_size=0.2, random_state=42):
    # Open the HDF5 file
    with h5py.File(filepath, "r") as f:
        X = np.array(f["images"])
        y = np.array(f["labels"])

    print("Dataset loaded:")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Unique labels:", np.unique(y))

    # Normalize pixel values (0–1 range)
    X = X.astype("float32") / 255.0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    load_binary_galaxy_dataset()
