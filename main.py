import os

from dotenv import load_dotenv

from src.selection import Selector
from src.preprocessing import Preprocessor
from src.transformation import Transformer
from src.mining import Miner
from src.evaluation import Evaluator

if __name__ == "__main__":
    text_column = "description"
    class_column = "class"

    load_dotenv()

    NUM_SAMPLES: int = int(os.environ.get("NUM_SAMPLES"))
    MAX_LEN: int = int(os.environ.get("MAX_LEN"))
    TEST_SIZE: float = float(os.environ.get("TEST_SIZE"))
    VOCAB_LENGTH: int = int(os.environ.get("VOCAB_LENGTH"))
    EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM"))
    INPUT_LENGTH: int = int(os.environ.get("INPUT_LENGTH"))
    DROP_RATE: float = float(os.environ.get("DROP_RATE"))
    RECURRENT_DROPOUT: float = float(os.environ.get("RECURRENT_DROPOUT"))
    EPOCHS: int = int(os.environ.get("EPOCHS"))
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE"))

    selector = Selector(text_column=text_column, class_column=class_column)
    df = selector.read_csv(path="data/news.psv")

    preprocessor = Preprocessor(text_column=text_column, class_column=class_column)
    df = preprocessor.remove_duplicated(df=df)
    df = preprocessor.remove_outliers(df=df, column=text_column)

    transformer = Transformer(text_column=text_column, class_column=class_column)
    df = transformer.to_lowercase(df=df)
    df = transformer.balance_class_amount(num_samples=NUM_SAMPLES, df=df)
    tokenizer = transformer.tokenization(df=df)
    X = transformer.pad_sequences(df=df, tokenizer=tokenizer, max_len=MAX_LEN)
    labels = transformer.extract_labels(df=df)
    y = transformer.encode_categories(df=df, labels=labels)

    miner = Miner(X=X, y=y)
    X_train, X_test, y_train, y_test = miner.train_test_split(test_size=TEST_SIZE)
    model = miner.build(
        X_train=X_train, labels=labels,
        vocab_length=VOCAB_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        input_length=INPUT_LENGTH,
        drop_rate=DROP_RATE,
        recurrent_dropout=RECURRENT_DROPOUT,
        activation="softmax",
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=3, batch_size=32)

    evaluator = Evaluator(X_test=X_test, y_test=y_test)
    y_pred = evaluator.predict(model=model, batch_size=BATCH_SIZE)
    df_cm = evaluator.generate_confusion_matrix(y_pred=y_pred, labels=labels)
    report = evaluator.generate_report(labels=labels)
