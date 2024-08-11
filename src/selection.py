import pandas as pd

class Selector:

    def __init__(self, text_column: str, class_column: str) -> None:
        self.text_column = text_column
        self.class_column = class_column

    def read_csv(self, path: str):

        df = pd.read_csv(
            filepath_or_buffer=path,
            sep="|"
        )
        return df
