import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from .config import CLASS_NAMES

def get_true_pred(model, generator):
    generator.reset()
    y_prob = model.predict(generator, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes
    return y_true, y_pred, y_prob

def evaluate_model(model, generator, model_name):
    y_true, y_pred, y_prob = get_true_pred(model, generator)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
        auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_weighted,
        "auc_macro": auc,
        "report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

def comparison_table(models, generators):
    rows = []
    for name, model in models.items():
        r = evaluate_model(model, generators[name], name)
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Macro Precision": r["macro_precision"],
            "Macro Recall": r["macro_recall"],
            "Macro F1": r["macro_f1"],
            "Weighted F1": r["weighted_f1"],
            "AUC (macro)": r["auc_macro"],
        })
    return pd.DataFrame(rows)
