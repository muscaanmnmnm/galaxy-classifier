import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import load_binary_galaxy_dataset
from model import build_galaxy_cnn

def train_model(epochs=10, batch_size=32):
    # Load dataset (currently gives train+test only)
    (X, y), (X_test, y_test) = load_binary_galaxy_dataset()

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Build model
    model = build_galaxy_cnn(input_shape=X_train.shape[1:], num_classes=2)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Save model
    model.save("models/galaxy_cnn.h5")
    print("✅ Model saved at models/galaxy_cnn.h5")

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"🎯 Test Accuracy: {test_acc:.4f}")

    return history

if __name__ == "__main__":
    train_model()
