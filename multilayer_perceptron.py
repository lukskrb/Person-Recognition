from matplotlib import pyplot as plt
from sklearn.calibration import label_binarize
import tensorflow as tf
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve
from data_manipulation import X, X_train, X_test, y_train, y_test

# Define number of classes
num_classes = len(np.unique(y_train))

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=25, batch_size=32)

# Evaluate accuracy
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# Predictions
y_pred_proba = model.predict(X_test)  # probabilities
y_pred = np.argmax(y_pred_proba, axis=1)  # class labels

# Metrics
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average=None)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print("F1 Score:", f1)
print("Recall:", recall)
print("ROC AUC:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')  # values_format='d' for integer counts
plt.title("Confusion Matrix - Neural")
plt.show()

# Find the largest off-diagonal values (most common confusions)
errors = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            errors.append((i, j, cm[i, j]))

# Sort by number of mistakes
errors_sorted = sorted(errors, key=lambda x: x[2], reverse=True)

print("Most frequent misclassifications:")
for true_class, pred_class, count in errors_sorted[:5]:  # top 5
    print(f"User {true_class} often confused with User {pred_class} ({count} times)")

# ROC curves
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score = model.predict(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc_dict = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_dict["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc_dict["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC (AUC = {roc_auc_dict["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=3)
plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC (AUC = {roc_auc_dict["macro"]:.2f})', color='navy', linestyle=':', linewidth=3)

colors = plt.cm.get_cmap('tab10', n_classes)
for i, color in zip(range(n_classes), colors.colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC - Multilayer perceptron')
plt.legend(loc="lower right")
plt.show()