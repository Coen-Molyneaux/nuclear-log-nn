from loader import DocumentLoader
import numpy as np
import tensorflow as tf

MODEL_PATH = '/home/coen-molyneaux/code/ml/nuclear-log-nn/src/model.keras'
CSV_FILE   = '/home/coen-molyneaux/code/ml/nuclear-log-nn/dataset/log.csv'

BATCH = 32

# ---- OPTIONAL: only needed if you normalized X manually during training ----
# If you used a Keras Normalization() layer inside the saved model, leave these as None.
X_MEAN_PATH = None  # e.g. '/home/.../x_mean.npy'
X_STD_PATH  = None  # e.g. '/home/.../x_std.npy'

# ---- OPTIONAL: only needed if you normalized y during training ----
Y_MEAN = None  # float
Y_STD  = None  # float


def main():
    loader = DocumentLoader(csvfile=CSV_FILE)
    x, y = loader.load()

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    N = len(x)
    assert len(x) == len(y), "X and y length mismatch"

    val_end = int(0.85 * N)
    x_test = x[val_end:]
    y_test = y[val_end:]

    if X_MEAN_PATH and X_STD_PATH:
        x_mean = np.load(X_MEAN_PATH)
        x_std  = np.load(X_STD_PATH)
        x_test = (x_test - x_mean) / (x_std + 1e-8)

    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH)

    results = model.evaluate(test_ds, verbose=1)
    print("evaluate() returned:", results)

    y_pred = model.predict(test_ds, verbose=0)

    if Y_MEAN is not None and Y_STD is not None:
        y_pred_real = y_pred * Y_STD + Y_MEAN
        y_test_real = y_test * Y_STD + Y_MEAN
        print("\nFirst 10 [actual, pred] in REAL units:")
        print(np.hstack([y_test_real[:10], y_pred_real[:10]]))
    else:
        print("\nFirst 10 [actual, pred] (already REAL units):")
        print(np.hstack([y_test[:10], y_pred[:10]]))


if __name__ == "__main__":
    main()