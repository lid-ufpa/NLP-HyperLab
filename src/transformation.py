from typing import List

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Transformation:
    def __init__(self, text_column: str, class_column: str) -> None:
        self.text_column = text_column
        self.class_column = class_column
    
    def to_lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = df["text"].str.lower()
        return df
    
    def balance_class_amount(self, num_samples: int, df: pd.DataFrame) -> pd.DataFrame:
        samples_by_class = [df.loc[df["class"] == label].sample(num_samples) for label in df[self.column].unique()]
        df = pd.concat(samples_by_class)
        df = df.sample(frac=1)
        df = df.reset_index(drop=True)
        return df
    
    def tokenization(self, df: pd.DataFrame) -> Tokenizer:
        tokenizer = Tokenizer(
            num_words=100000,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            char_level=False,
        )

        tokenizer.fit_on_texts(df[self.text_column])

        return tokenizer
    
    def pad_sequences(self, df: pd.DataFrame, tokenizer: Tokenizer, max_len: int) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(df[self.text_column].values)

        X = pad_sequences(
            sequences=sequences,
            maxlen=max_len,
            padding="pre",
            truncating="post",
        )

        return X
    
    def extract_labels(self, df: pd.DataFrame) -> List[str]:
        unique_classes = df[self.class_column].unique()
        labels = sorted(unique_classes)
        return labels
    
    def encode_categories(self, df: pd.DataFrame, labels: List[str]) -> np.ndarray:
        category_type = pd.CategoricalDtype(categories=labels)
        classes_as_categories = df["class"].astype(category_type)
        dummies = pd.get_dummies(classes_as_categories)
        y = dummies.values
        return y
    