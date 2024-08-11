from typing import List

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    LSTM,
    Dense,
    Flatten
)

class Modeling:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def train_test_split(self, test_size: float):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=42
        )

        return X_train, X_test, y_train, y_test
    
    def build(
            self, X_train: np.ndarray, labels: np.ndarray,
            vocab_length: int, embedding_dim: int, input_length: int,
            drop_rate: float, recurrent_dropout: float,
            activation: str, loss: str, optimizer: str, metrics: List[str]
        ) -> Model:
        _input: Input = Input(shape=(X_train.shape[1],))

        embedding: Embedding = Embedding(
            input_dim=vocab_length,
            output_dim=embedding_dim,
            input_length=input_length
        )

        x = embedding(_input)

        x = SpatialDropout1D(rate=drop_rate)(x)

        x = LSTM(embedding_dim, dropout=drop_rate, recurrent_dropout=recurrent_dropout)(x)

        x = Dense(len(labels), activation=activation)(x)

        output = Flatten()(x)

        model = Model(_input, output)

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )

        return model
    
    def fit(self, model, X_train, y_train, epochs: int, batch_size: int):
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size
        )

        return model
    