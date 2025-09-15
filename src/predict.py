import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_binary_galaxy_dataset

# Load model
model = tf.keras.models.load_model("models/galaxy_cnn.h5")

# Load dataset (for testing)
(_, _), (X_test, y_test) = load_binary_galaxy_dataset()

# Pick one random test sample
idx = np.random.randint(0, len(X_test))
sample = X_test[idx]
true_label = y_test[idx]

# Predict
pred = model.predict(sample[np.newaxis, ...], verbose=0)
pred_class = np.argmax(pred)

# Labels (double-check mapping)
labels = ["Elliptical", "Spiral"]

# Show image + prediction
plt.imshow(sample)
plt.title(f"True: {labels[true_label]} | Predicted: {labels[pred_class]}")
plt.axis("off")
plt.show()
