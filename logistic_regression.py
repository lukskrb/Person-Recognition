from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)  # increase max_iter to avoid convergence issues
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average=None)
roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr')

print("Accuracy (Top-1):", acc)
print("F1 score (weighted):", f1)
print("Recall score (per class):", recall)
print("ROC AUC (OvR):", roc_auc)

# Top-k accuracy
top3_acc = top_k_accuracy_score(y_test, y_score, k=3)
print("Top-3 Accuracy:", top3_acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Find most frequent misclassifications
errors = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            errors.append((i, j, cm[i, j]))
errors_sorted = sorted(errors, key=lambda x: x[2], reverse=True)

print("Most frequent misclassifications:")
for true_class, pred_class, count in errors_sorted[:5]:
    print(f"User {true_class} often confused with User {pred_class} ({count} times)")

# Prepare for ROC and PR
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

# ROC Curves
fpr, tpr, roc_auc = {}, {}, {}
for i, cls in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(classes)
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC
plt.figure(figsize=(8, 6))
for i, cls in enumerate(classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {cls} (AUC = {roc_auc[i]:.2f})')
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
         color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Logistic Regression")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curves
precision, recall, pr_auc = {}, {}, {}
for i, cls in enumerate(classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

# Micro-average
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
pr_auc["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

# Macro-average
pr_auc["macro"] = average_precision_score(y_test_bin, y_score, average="macro")

# Plot PR
plt.figure(figsize=(8, 6))
for i, cls in enumerate(classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {cls} (AP = {pr_auc[i]:.2f})')
plt.plot(recall["micro"], precision["micro"],
         label=f'Micro-average (AP = {pr_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()

print("Average Precision (AP) / PR AUC scores:")
print("Micro-average:", pr_auc["micro"])
print("Macro-average:", pr_auc["macro"])
