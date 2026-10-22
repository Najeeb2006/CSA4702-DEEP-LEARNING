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
