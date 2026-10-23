"""
Deep Learning Application in Data Science
Task: Classify handwritten digits (0-9) with a Convolutional Neural Network.
Dataset: MNIST via TensorFlow/Keras.
Outputs: metrics, training curves, confusion matrix, sample predictions.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
OUTPUT_DIR = Path("deep_learning_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

keras.utils.set_random_seed(SEED)

def main():
    # 1. Load public dataset
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    print("Full training set:", x_train_full.shape, y_train_full.shape)
    print("Test set:", x_test.shape, y_test.shape)

    # 2. Create validation split from training data only
    x_train = x_train_full[:-6000]
    y_train = y_train_full[:-6000]
    x_val = x_train_full[-6000:]
    y_val = y_train_full[-6000:]

    # 3. Preprocess: float32 and normalize pixel intensity from [0,255] to [0,1].
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_val = (x_val.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    # 4. CNN architecture
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 5. Training with early stopping and best-checkpoint restoration
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "best_mnist_cnn.keras"),
            monitor="val_loss", save_best_only=True
        )
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2
    )

    # 6. Final evaluation on untouched test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))
    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.4f}\\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\\n\\n")
        f.write(classification_report(y_test, predictions, digits=4))

    # 7. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="Training")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Training")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    # 8. Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title("MNIST Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 9. Save a small sample of predictions for inspection
    np.savez_compressed(
        OUTPUT_DIR / "sample_predictions.npz",
        images=x_test[:100], actual=y_test[:100],
        predicted=predictions[:100], probabilities=probabilities[:100]
    )
    print(f"\\nSaved metrics and plots to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
