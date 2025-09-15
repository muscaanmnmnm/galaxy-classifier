import h5py
import numpy as np
from tensorflow import keras

# Load dataset
file_path = "data/archive/Binary_2_5_dataset.h5"
with h5py.File(file_path, "r") as f:
    X = np.array(f["images"])
    y = np.array(f["labels"])

print("✅ Dataset loaded:", X.shape, y.shape)
print("Unique labels in dataset:", np.unique(y))

# Load trained model
model = keras.models.load_model("models/galaxy_cnn.h5")


# Test 20 random samples
indices = np.random.choice(len(X), 20, replace=False)
label_map = {0: "elliptical", 1: "spiral"}

for idx in indices:
    img = X[idx]
    true_label = y[idx]

    img_input = np.expand_dims(img, axis=0)  # shape (1, 256, 256, 3)
    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction)

    print(f"Image {idx}: True = {label_map[true_label]}, Predicted = {label_map[predicted_class]}, Raw = {prediction}")
