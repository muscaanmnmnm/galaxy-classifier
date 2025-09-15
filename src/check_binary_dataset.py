import h5py

file_path = "data/archive/Binary_2_5_dataset.h5"

with h5py.File(file_path, "r") as f:
    print("Keys in file:", list(f.keys()))
    for key in f.keys():
        print(f"{key} shape:", f[key].shape)
