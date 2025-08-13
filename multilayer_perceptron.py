import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score
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
recall = recall_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print("F1 Score:", f1)
print("Recall:", recall)
print("ROC AUC:", roc_auc)
