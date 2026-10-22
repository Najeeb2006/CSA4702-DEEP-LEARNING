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
