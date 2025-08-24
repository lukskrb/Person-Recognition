from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score, roc_auc_score
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
