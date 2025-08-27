from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Train Gradient Boosting classifier
# -------------------------
clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
).fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = clf.predict(X_test)

# -------------------------
# Metrics
# -------------------------
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average=None)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", acc)
print("F1 score:", f1)
print("Recall score:", recall)
print("ROC AUC: ", roc_auc)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()

# -------------------------
# Most frequent misclassifications
# -------------------------
errors = [(i, j, cm[i, j]) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
errors_sorted = sorted(errors, key=lambda x: x[2], reverse=True)

print("Most frequent misclassifications:")
for true_class, pred_class, count in errors_sorted[:5]:
    print(f"User {true_class} often confused with User {pred_class} ({count} times)")

# Feature importances
importances = clf.feature_importances_
print("Feature importances:", importances)

# ROC curves 
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predict probabilities
y_score = clf.predict_proba(X_test)

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
plt.title('Multi-class ROC - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()
