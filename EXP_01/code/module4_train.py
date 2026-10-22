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
