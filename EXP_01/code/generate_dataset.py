# generate_dataset.py : builds the 500-sample synthetic shape-image dataset
# 5 classes x 100 images, each 16x16 grayscale, stored one image per CSV row
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_PER_CLASS, SIZE = 100, 16
CLASSES = ["Circle", "Square", "Triangle", "Cross", "Diagonal"]
yy, xx = np.mgrid[0:SIZE, 0:SIZE]

def draw(cls, cx, cy, r, angle):
    img = np.zeros((SIZE, SIZE))
    dx, dy = xx - cx, yy - cy
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca      # rotated coordinates
    if cls == 0:   img[(dx**2 + dy**2) <= r**2] = 1                      # filled circle
    elif cls == 1: img[(np.abs(u) <= r*0.85) & (np.abs(v) <= r*0.85)] = 1  # square
    elif cls == 2: img[(v >= -r) & (v <= r) & (np.abs(u) <= (r - v) / 2)] = 1  # triangle
    elif cls == 3: img[((np.abs(u) <= r) & (np.abs(v) <= 1.0)) | ((np.abs(v) <= r) & (np.abs(u) <= 1.0))] = 1  # cross
    else:          img[(np.abs(v) <= 1.1) & (np.abs(u) <= r*1.3)] = 1    # bar / diagonal line
    return img

rows = []
for c in range(len(CLASSES)):
    for k in range(N_PER_CLASS):
        r = rng.uniform(3.5, 5.5)
        cx, cy = rng.uniform(6, 10, 2)
        if c == 4:   # diagonal bar: leaning to the left or to the right
            angle = rng.uniform(np.pi/6, np.pi/3) if rng.random() < 0.5 else rng.uniform(2*np.pi/3, 5*np.pi/6)
        elif c == 0: angle = 0.0
        else:        angle = rng.uniform(-np.pi/6, np.pi/6)   # small random rotation
        intensity = rng.uniform(0.65, 1.0)
        noise = rng.uniform(0.03, 0.15)
        img = draw(c, cx, cy, r, angle) * intensity
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
        gx, gy = np.gradient(img)
        rows.append({
            "image_id": f"IMG{len(rows)+1:04d}", "class_id": c, "class_name": CLASSES[c],
            "shape_size": round(r, 3), "center_x": round(cx, 3), "center_y": round(cy, 3),
            "rotation_deg": round(np.degrees(angle), 2), "intensity": round(intensity, 3),
            "noise_sigma": round(noise, 3), "mean_brightness": round(img.mean(), 4),
            "contrast_std": round(img.std(), 4),
            "edge_density": round(np.hypot(gx, gy).mean(), 4),
            "foreground_ratio": round((img > 0.5).mean(), 4),
            **{f"px_{i:03d}": round(v, 4) for i, v in enumerate(img.ravel())}})

df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("../dataset/shape_images_500.csv", index=False)
print(df.shape)
