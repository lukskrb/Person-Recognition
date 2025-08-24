from matplotlib import pyplot as plt
from data_manipulation import X_train, X_test, y_train, y_test
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np

# -------------------------
# 1. Definiraj modele
# -------------------------
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "GradBoost": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=0)
}

results = {}
recall_df = pd.DataFrame()

# -------------------------
# 2. Treniraj i evaluiraj svaki model
# -------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall_per_user = recall_score(y_test, y_pred, average=None)
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)
    
    results[name] = {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "recall_per_user": recall_per_user
    }
    
    recall_df[name] = recall_per_user

# -------------------------
# 3. Dodaj kolonu korisnika
# -------------------------
recall_df["User"] = np.unique(y_test)
recall_df = recall_df[["User"] + list(models.keys())]

print("\nTablica recall-a po korisniku:")
print(recall_df)

# -------------------------
# 4. Grafikon recall-a po korisniku
# -------------------------
bar_width = 0.13
x = np.arange(len(recall_df["User"]))

plt.figure(figsize=(14,6))
for i, model in enumerate(models.keys()):
    plt.bar(x + i*bar_width, recall_df[model], width=bar_width, label=model)

plt.xticks(x + bar_width*2.5, recall_df["User"])
plt.xlabel("User")
plt.ylabel("Recall")
plt.title("Recall per User")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 5. Export u LaTeX tablicu
# -------------------------
latex_table = recall_df.to_latex(index=False, float_format="%.3f")
print("\nLaTeX kod za tablicu:\n")
print(latex_table)

with open("recall_per_user_table.tex", "w") as f:
    f.write(latex_table)
