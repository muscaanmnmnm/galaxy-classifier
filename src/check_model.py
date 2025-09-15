import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("models/galaxy_cnn.h5")

# Print summary
model.summary()

# Print input shape
print("Model input shape:", model.input_shape)
