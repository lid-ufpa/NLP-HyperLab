import statistics

import pandas as pd

class Preprocessor:
    def __init__(self, text_column: str, class_column: str) -> None:
        self.text_column = text_column
        self.class_column = class_column

    def remove_duplicated(self, df: pd.DataFrame):
        df = df.drop_duplicates(subset=self.text_column)
        return df

    def check_outlier(self, value, quantiles, amplitude):
        if (value < quantiles[0] - (3 * amplitude)) or (value > quantiles[2] + (3 * amplitude)):
            return True
        return False

    def remove_outliers(self, df: pd.DataFrame, column: str):

        df[f"len{column}"] = df[column].str.len()

        quantiles = statistics.quantiles(df[column])

        amplitude = quantiles[2] - quantiles[0]

        df[f"outlier_{column}"] = df[column].map(lambda x: self.check_outlier(
            value=x,
            quantiles=quantiles,
            amplitude=amplitude
        ))

        df = df.loc[~(df[f"outlier_{column}"])]
        df = df.drop(columns=[f"len{column}", f"outlier_{column}"])

        return df
