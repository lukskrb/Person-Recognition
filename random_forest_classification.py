from data_manipulation import X, y
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
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

print("Accuracies:", accuracies)
print("Average Accuracy:", np.mean(accuracies))
print("Average ROC AUC:", np.mean(roc_aucs))
print("Average F1 Score:", np.mean(f1_scores))
print("Average Recall:", np.mean(recalls))
