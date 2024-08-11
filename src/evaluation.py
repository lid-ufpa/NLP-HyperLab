from typing import Dict, List

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
    
    def get_binary_cms(self) -> Dict[str, pd.DataFrame]:
        """Given n labels, generates n 2x2 matrices describing TP, TN, FP, FN of a multiclass confusion matrix."""

        binary_cms = {}

        for label in self.labels:
            # getting the index of label in evaluation list of labels
            i = self.labels.index(label)

            # calculating metrics
            TP = self.cm.iloc[i, i]
            FP = self.cm.iloc[:, i].sum().sum() - TP
            FN = self.cm.iloc[i, :].sum().sum() - TP
            TN = self.cm.sum().sum() - TP - FP - FN

            # construct DataFrame for current label
            binary_cms[label] = pd.DataFrame(
                data={
                    "PREDICTED POSITIVE": [TP, FP],
                    "PREDICTED NEGATIVE": [FN, TN]
                },
                index=["ACTUAL POSITIVE", "ACTUAL NEGATIVE"]
            )

        return binary_cms
    
    def report(self, label: str) -> dict:
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

        binary_cm: pd.DataFrame = self.binary_cms[label]

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
            "FNR": FNR, "FPR": FPR,
            "FDR": FDR, "FOR": FOR,
            "PLR": PLR, "NLR": NLR,
            "PT": PT, "TS": TS,
            "PRE": PRE, "ACC": ACC,
            "BA": BA, "F1": F1
        }

    def generate_report(self) -> pd.DataFrame:
        """Generate complete evaluation report."""

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

        reports: dict = {}

        for i in range(len(self.labels)):
            label: str = self.labels[i]
            reports[label]: dict = self.extract_metrics(label)

        general_report: dict = {"label": self.labels}

        for metric in metrics:
            general_report[metric]: list = [reports[label][metric] for label in reports.keys()]

        return pd.DataFrame(general_report).fillna(0)