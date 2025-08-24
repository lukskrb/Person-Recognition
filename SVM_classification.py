from matplotlib import pyplot as plt
from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

# Kreiranje SVM klasifikatora s omogućenom procjenom vjerojatnosti
svm = SVC(probability=True)   
svm.fit(X_train, y_train)

# Predikcija
y_pred = svm.predict(X_test)

# Evaluacija
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average=None)

print("Accuracy:", acc)
print("F1 score:", f1)
print("Recall score:", recall)
print("ROC AUC: ", roc_auc_score(y_test, svm.predict_proba(X_test), multi_class='ovr'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')  # values_format='d' for integer counts
plt.title("Confusion Matrix - SVM")
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