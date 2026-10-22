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
