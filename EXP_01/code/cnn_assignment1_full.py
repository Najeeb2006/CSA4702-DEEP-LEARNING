# Assignment 1: CNN Design and Gradient-Based Training

# %%
# Module 1: Import Libraries, Load Dataset and Preprocess
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Dataset: 500 grayscale images (16 x 16) stored one image per row, 269 fields
DATA_PATH = os.path.join("..", "dataset", "shape_images_500.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "shape_images_500.csv"
df = pd.read_csv(DATA_PATH)

pixel_cols = [c for c in df.columns if c.startswith("px_")]
meta_cols = [c for c in df.columns if c not in pixel_cols]
print("Dataset shape (rows, fields):", df.shape)
print("Pixel fields    :", len(pixel_cols))
print("Metadata fields :", len(meta_cols))
print("Missing values  :", int(df.isnull().sum().sum()))
print("\nClass distribution:")
print(df["class_name"].value_counts().sort_index().to_string())
print("\nFirst 5 records (selected fields):")
show = ["image_id", "class_name", "shape_size", "rotation_deg", "intensity",
        "noise_sigma", "edge_density", "px_000", "px_120", "px_255"]
print(df[show].head().to_string(index=False))

# Table 1: description of the dataset fields
table1 = pd.DataFrame({
    "Field Group": ["Identifier", "Label", "Shape geometry", "Appearance",
                    "Image statistics", "Pixel values", "Total"],
    "Fields": ["image_id", "class_id, class_name",
               "shape_size, center_x, center_y, rotation_deg",
               "intensity, noise_sigma",
               "mean_brightness, contrast_std, edge_density, foreground_ratio",
               "px_000 to px_255 (16 x 16 image, row-major)", "-"],
    "Count": [1, 2, 4, 2, 4, 256, len(df.columns)]})
print("\nTable 1: Dataset description")
print(table1.to_string(index=False))

# Preprocessing: pixels are already in [0, 1]; reshape to (N, H, W, C)
X = df[pixel_cols].values.astype("float32").reshape(-1, 16, 16, 1)
y = df["class_id"].values.astype("int32")
class_names = df.sort_values("class_id")["class_name"].unique().tolist()

# Stratified split: 70 % train, 15 % validation, 15 % test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("\nPixel range     :", X.min(), "to", X.max())
print("Train / Val / Test :", X_train.shape, X_val.shape, X_test.shape)

# Plot 1: sample images of every class
fig, axes = plt.subplots(3, 5, figsize=(9, 5.8))
for c in range(5):
    idx = np.where(y == c)[0][:3]
    for r, i in enumerate(idx):
        axes[r, c].imshow(X[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(class_names[c], fontsize=11)
fig.suptitle("Plot 1: Sample Images from Each Class (16 x 16, grayscale)", fontsize=12)
plt.tight_layout()
plt.show()


# %%
# Module 2: Convolution, Stride, Padding and Pooling (manual implementation)
def out_size(n, k, s, p):
    """Output size of a convolution: floor((n + 2p - k) / s) + 1"""
    return (n + 2 * p - k) // s + 1

def conv2d_manual(img, kernel, stride=1, padding=0):
    img_p = np.pad(img, padding)                 # zero padding on all four sides
    k = kernel.shape[0]
    n_out = out_size(img.shape[0], k, stride, padding)
    out = np.zeros((n_out, n_out))
    for i in range(n_out):
        for j in range(n_out):
            patch = img_p[i * stride:i * stride + k, j * stride:j * stride + k]
            out[i, j] = np.sum(patch * kernel)   # element-wise product and sum
    return out

sample = X_train[0, :, :, 0]
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")
fmap = conv2d_manual(sample, sobel_x, stride=1, padding=0)
print("Input image shape      :", sample.shape)
print("Kernel (Sobel-x) shape :", sobel_x.shape)
print("Feature map shape      :", fmap.shape)

# Verify the manual convolution against the Keras Conv2D layer
conv = layers.Conv2D(1, 3, padding="valid", use_bias=False)
conv.build((None, 16, 16, 1))
conv.set_weights([sobel_x.reshape(3, 3, 1, 1)])
keras_map = conv(sample.reshape(1, 16, 16, 1)).numpy()[0, :, :, 0]
print("Max |manual - Keras|   :", float(np.abs(fmap - keras_map).max()))

# Effect of kernel size (k), stride (s) and padding (p) on a 16 x 16 input
rows = []
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (5, 1, 0), (5, 2, 2), (7, 1, 0)]:
    formula = out_size(16, k, s, p)
    manual = conv2d_manual(sample, np.ones((k, k), dtype="float32"), s, p).shape[0]
    padded = layers.ZeroPadding2D(p)(tf.zeros((1, 16, 16, 1)))
    actual = layers.Conv2D(1, k, strides=s)(padded).shape[1]
    rows.append([f"{k} x {k}", s, p, f"{formula} x {formula}",
                 f"{manual} x {manual}", f"{actual} x {actual}",
                 "Yes" if formula == manual == actual else "No"])
table2 = pd.DataFrame(rows, columns=["Kernel", "Stride", "Padding", "Formula Output",
                                     "Manual Output", "Keras Output", "Match"])
print("\nTable 2: Convolution output size for a 16 x 16 input")
print(table2.to_string(index=False))

# Pooling: 2 x 2 max pooling with stride 2 halves the spatial size
relu_map = np.maximum(fmap, 0)                       # 14 x 14 feature map after ReLU
pooled = relu_map.reshape(7, 2, 7, 2).max(axis=(1, 3))
print("\nBefore pooling :", relu_map.shape, " After 2x2 max-pool :", pooled.shape)
print("Top-left 2x2 block:\n", np.round(relu_map[:2, :2], 3))
print("Max-pooled value  :", round(float(pooled[0, 0]), 3))


# %%
# Module 3: Design the CNN (initialisation, output shapes, parameter counts)
def build_cnn(l2=1e-3, dropout=0.3, seed=SEED):
    keras.utils.set_random_seed(seed)
    he = lambda: keras.initializers.HeNormal(seed=seed)          # He initialisation for ReLU
    reg = regularizers.l2(l2) if l2 > 0 else None                # L2 weight decay
    return keras.Sequential([
        layers.Input(shape=(16, 16, 1)),
        layers.Conv2D(16, 3, strides=1, padding="same", activation="relu",
                      kernel_initializer=he(), name="conv1"),
        layers.MaxPooling2D(2, name="pool1"),
        layers.Conv2D(32, 3, strides=1, padding="valid", activation="relu",
                      kernel_initializer=he(), name="conv2"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", kernel_initializer=he(),
                     kernel_regularizer=reg, name="fc1"),
        layers.Dropout(dropout, name="dropout"),
        layers.Dense(5, activation="softmax",
                     kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
                     kernel_regularizer=reg, name="output")], name="shape_cnn")

model = build_cnn()
model.summary()

# Verify output shapes and parameter counts with the formulas
rows, total, c_in = [], 0, 1
for layer in model.layers:
    shape = tuple(layer.output.shape[1:])
    if isinstance(layer, layers.Conv2D):
        k, f = layer.kernel_size[0], layer.filters
        manual = (k * k * c_in + 1) * f
        formula = f"({k}x{k}x{c_in}+1) x {f}"
        c_in = f
    elif isinstance(layer, layers.Dense):
        n_in = int(layer.input.shape[-1])
        manual = (n_in + 1) * layer.units
        formula = f"({n_in}+1) x {layer.units}"
    else:
        manual, formula = 0, "-"
    assert manual == layer.count_params(), layer.name
    total += manual
    rows.append([layer.name, layer.__class__.__name__,
                 " x ".join(map(str, shape)), formula, manual])
rows.append(["Total", "", "", "", total])
table3 = pd.DataFrame(rows, columns=["Layer", "Type", "Output Shape", "Parameter Formula", "Parameters"])
print("\nTable 3: Layer-wise output shape and parameter count")
print(table3.to_string(index=False))
print("\nManual total:", total, "| Keras total:", model.count_params())

# Weight initialisation check: He normal gives std = sqrt(2 / fan_in)
print("\nInitialisation check (He normal):")
for name, fan_in in [("conv2", 3 * 3 * 16), ("fc1", 288)]:
    w = model.get_layer(name).get_weights()[0]
    print(f"  {name:<6} fan_in={fan_in:<4} expected std={np.sqrt(2 / fan_in):.4f}"
          f"  actual std={w.std():.4f}  bias all zero: {bool(np.all(model.get_layer(name).get_weights()[1] == 0))}")


# %%
# Module 4: Gradient-Based Training (Adam optimiser, L2 + Dropout regularisation)
EPOCHS, BATCH, LR = 40, 32, 1e-3
model = build_cnn(l2=1e-3, dropout=0.3)

# Gradients of the loss with respect to every weight for one mini-batch (before training)
xb, yb = X_train[:BATCH], y_train[:BATCH]
with tf.GradientTape() as tape:
    probs = model(xb, training=True)
    loss = keras.losses.sparse_categorical_crossentropy(yb, probs)
    loss = tf.reduce_mean(loss) + tf.add_n(model.losses)
grads = tape.gradient(loss, model.trainable_variables)
print(f"Initial mini-batch loss: {float(loss):.4f}   (untrained network; uniform guess = ln 5 = {np.log(5):.4f})")
print("Gradient L2-norm per parameter tensor:")
for v, g in zip(model.trainable_variables, grads):
    print(f"  {v.path.replace("shape_cnn/", ""):<14} shape={str(tuple(v.shape)):<16} ||grad|| = {float(tf.norm(g)):.4f}")

# Training configuration
print(f"\nOptimiser : Adam (lr={LR})   Loss : sparse categorical cross-entropy")
print(f"Epochs    : {EPOCHS}   Batch size : {BATCH}   L2 : 0.001   Dropout : 0.3")

model.compile(optimizer=keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
log = keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: print(
    f"Epoch {e + 1:>2}/{EPOCHS}  loss={l['loss']:.4f}  acc={l['accuracy']:.4f}  "
    f"val_loss={l['val_loss']:.4f}  val_acc={l['val_accuracy']:.4f}")
    if e == 0 or (e + 1) % 5 == 0 else None)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[log])
h = history.history
print(f"\nFinal train accuracy: {h['accuracy'][-1]:.4f} | final validation accuracy: {h['val_accuracy'][-1]:.4f}")

# Plot 2: learning curves
ep = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, h["loss"], color="#2E7D32", label="Training loss")
ax[0].plot(ep, h["val_loss"], color="#E65100", label="Validation loss")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss")
ax[1].plot(ep, h["accuracy"], color="#2E7D32", label="Training accuracy")
ax[1].plot(ep, h["val_accuracy"], color="#E65100", label="Validation accuracy")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.suptitle("Plot 2: Learning Curves of the CNN (Adam, lr = 0.001)", fontsize=12)
plt.tight_layout()
plt.show()


# %%
# Module 5: Evaluation, Optimiser and Regularisation Comparison
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Compare optimisers and the effect of regularisation (same seed, same data)
def run(name, optimizer, lr, l2, dropout):
    m = build_cnn(l2, dropout)
    m.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hh = m.fit(X_train, y_train, validation_data=(X_val, y_val),
               epochs=EPOCHS, batch_size=BATCH, verbose=0).history
    tr, va = hh["accuracy"][-1], hh["val_accuracy"][-1]
    te = m.evaluate(X_test, y_test, verbose=0)[1]
    return [name, lr, l2, dropout, round(tr, 4), round(va, 4), round(te, 4), round(tr - va, 4)]

results = [
    run("SGD", keras.optimizers.SGD(0.01), 0.01, 1e-3, 0.3),
    run("SGD + Momentum (0.9)", keras.optimizers.SGD(0.01, momentum=0.9), 0.01, 1e-3, 0.3),
    ["Adam", 0.001, 1e-3, 0.3, round(h["accuracy"][-1], 4), round(h["val_accuracy"][-1], 4),
     round(test_acc, 4), round(h["accuracy"][-1] - h["val_accuracy"][-1], 4)],   # model from Module 4
    run("Adam, no regularisation", keras.optimizers.Adam(1e-3), 0.001, 0.0, 0.0)]
table4 = pd.DataFrame(results, columns=["Configuration", "Learning Rate", "L2", "Dropout",
                                        "Train Acc", "Val Acc", "Test Acc", "Train - Val Gap"])
print(f"\nTable 4: Optimiser and regularisation comparison ({EPOCHS} epochs)")
print(table4.to_string(index=False))

# Plot 3: confusion matrix of the Adam model on the test set
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="YlOrBr")
ax.set_xticks(range(5)); ax.set_xticklabels(class_names, rotation=30)
ax.set_yticks(range(5)); ax.set_yticklabels(class_names)
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("Plot 3: Confusion Matrix on the Test Set")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
