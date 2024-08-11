from typing import Dict, List, Union

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from sklearn.metrics import confusion_matrix


class Evaluator:
    def __init__(self, X_test: np.ndarray, y_test: np.ndarray):
        self.X_test = X_test
        self.y_test = y_test

    def predict(self, model: Model, batch_size: int) -> np.ndarray:
        return model.predict(
            x=self.X_test,
            batch_size=batch_size,
            verbose=1
        )
    
    def generate_confusion_matrix(self, y_pred: np.ndarray, labels: List[str]) -> np.ndarray:
        cm = confusion_matrix(
            y_true=self.y_test.argmax(axis=1),
            y_pred=y_pred.argmax(axis=1),
        )

        return pd.DataFrame(cm, index=labels, columns=labels)
    
    def get_binary_cms(self, labels: List[str], df_cm: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Given n labels, generates n 2x2 matrices describing TP, TN, FP, FN of a multiclass confusion matrix."""

        binary_cms = {}

        for label in labels:
            # getting the index of label in evaluation list of labels
            i = labels.index(label)

            # calculating metrics
            TP = df_cm.iloc[i, i]
            FP = df_cm.iloc[:, i].sum().sum() - TP
            FN = df_cm.iloc[i, :].sum().sum() - TP
            TN = df_cm.sum().sum() - TP - FP - FN

            # construct DataFrame for current label
            binary_cms[label] = pd.DataFrame(
                data={
                    "PREDICTED POSITIVE": [TP, FP],
                    "PREDICTED NEGATIVE": [FN, TN]
                },
                index=["ACTUAL POSITIVE", "ACTUAL NEGATIVE"]
            )

        return binary_cms
    
    def compute_metrics(self, label: str, binary_cm: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Returns evaluation metrics for a given label.\n\n
        P: Positives;\n
        N: Negatives;\n
        TP: True Positives;\n
        FN: False Negatives;\n
        TN: True Negatives;\n
        FP: False Positives;\n
        TPR: True Positive Rate, Sensitivity, Recall, or Hit Rate;\n
        TNR: True Negative Rate, Specificity, Selectivity;\n
        PPV: Positive Predictive Value, or Precision;\n
        NPV: Negative Predictive Value;\n
        ACC: Accuracy;\n
        BA: Balanced Accuracy;\n
        F1: F1 Score.
        """

        P: int = binary_cm.loc["ACTUAL POSITIVE"].sum()
        N: int = binary_cm.loc["ACTUAL NEGATIVE"].sum()
        TP: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED POSITIVE"]
        FN: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED NEGATIVE"]
        FP: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED POSITIVE"]
        TN: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED NEGATIVE"]
        TPR: float = TP / P
        TNR: float = TN / N
        PPV: float = TP / (TP + FP)
        NPV: float = TN / (TN + FN)
        ACC: float = (TP + TN) / (P + N)
        BA: float = (TPR + TNR) / 2
        try:
            F1: float = 2 * (PPV * TPR) / (PPV + TPR)
        except ZeroDivisionError:
            F1 = 0

        return {
            "P": P, "N": N,
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "TPR": TPR, "TNR": TNR,
            "PPV": PPV, "NPV": NPV,
            "ACC": ACC, "BA": BA, "F1": F1
        }

    def generate_report(self, labels: List[str], df_cm: pd.DataFrame) -> pd.DataFrame:
        """Generate complete evaluation report."""

        binary_cms: Dict[str, pd.DataFrame] = self.get_binary_cms(
            labels=labels,
            df_cm=df_cm
        )

        metrics: list = [
            "P",
            "N",
            "TP",
            "FN",
            "FP",
            "TN",
            "TPR",
            "TNR",
            "PPV",
            "NPV",
            "ACC",
            "BA",
            "F1",
        ]

        reports: Dict[str, Union[int, float]] = {}

        for label in labels:
            binary_cm = binary_cms[label]
            reports[label] = self.compute_metrics(label, binary_cm)

        general_report: Dict = {"label": labels}

        for metric in metrics:
            general_report[metric] = [reports[label][metric] for label in reports.keys()]

        return pd.DataFrame(general_report).fillna(0)
    