from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("Accuracy:", accuracy)
print("ROC_AUC:", roc_auc)
print(classification_report(y_test, y_pred))
