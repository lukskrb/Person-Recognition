from matplotlib import pyplot as plt
from sklearn.calibration import label_binarize
from data_manipulation import X, y
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, roc_auc_score, f1_score, recall_score, roc_curve
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
roc_aucs = []
f1_scores = []
recalls = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    f1 = f1_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    accuracies.append(acc)
    roc_aucs.append(roc_auc)
    f1_scores.append(f1)
    recalls.append(rec)
    # Predict probabilities
    y_score = model.predict_proba(X_test)

print("Accuracies:", accuracies)
print("Average Accuracy:", np.mean(accuracies))
print("Average ROC AUC:", np.mean(roc_aucs))
print("Average F1 Score:", np.mean(f1_scores))
print("Average Recall:", np.mean(recalls))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')  # values_format='d' for integer counts
plt.title("Confusion Matrix - Random Forest")
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
plt.title('Multi-class ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()