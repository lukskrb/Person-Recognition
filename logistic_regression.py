from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average=None)

print("Accuracy:", acc)
print("F1 score:", f1)
print("Recall score:", recall)
print("ROC AUC: ", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))