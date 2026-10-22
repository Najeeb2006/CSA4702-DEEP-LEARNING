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
