from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y_true, name=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Patron+", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Patron+", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Patron+", zero_division=0)

    print(f"\n🧪 Evaluation for {name}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

    cm = confusion_matrix(y_true, y_pred, labels=["Under-Patron", "Patron+"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Under-Patron", "Patron+"], yticklabels=["Under-Patron", "Patron+"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true.map({"Under-Patron": 0, "Patron+": 1}), y_prob)
        auc_score = roc_auc_score(y_true.map({"Under-Patron": 0, "Patron+": 1}), y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
