import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_binary_galaxy_dataset

# Load trained model
model = tf.keras.models.load_model("models/galaxy_cnn.h5")

# Load dataset
(_, _), (X_test, y_test) = load_binary_galaxy_dataset()

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Non-Spiral", "Spiral"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Spiral", "Spiral"],
            yticklabels=["Non-Spiral", "Spiral"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
