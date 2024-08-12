from src.selection import Selector
from src.preprocessing import Preprocessor
from src.transformation import Transformer
from src.mining import Miner
from src.evaluation import Evaluator

def lambda_handler(event, context):

    NUM_SAMPLES = event["NUM_SAMPLES"]
    MAX_LEN = event["MAX_LEN"]
    TEST_SIZE = event["TEST_SIZE"]
    VOCAB_LENGTH = event["VOCAB_LENGTH"]
    EMBEDDING_DIM = event["EMBEDDING_DIM"]
    INPUT_LENGTH = event["INPUT_LENGTH"]
    DROP_RATE = event["DROP_RATE"]
    RECURRENT_DROPOUT = event["RECURRENT_DROPOUT"]
    EPOCHS = event["EPOCHS"]
    BATCH_SIZE = event["BATCH_SIZE"]

    result = execute_mining_steps(NUM_SAMPLES, MAX_LEN, TEST_SIZE, VOCAB_LENGTH, EMBEDDING_DIM, INPUT_LENGTH, DROP_RATE, RECURRENT_DROPOUT, EPOCHS, BATCH_SIZE)

    message = f"""
    NUM_SAMPLES: {NUM_SAMPLES}
    MAX_LEN: {MAX_LEN}
    TEST_SIZE: {TEST_SIZE}
    VOCAB_LENGTH: {VOCAB_LENGTH}
    EMBEDDING_DIM: {EMBEDDING_DIM}
    INPUT_LENGTH: {INPUT_LENGTH}
    DROP_RATE: {DROP_RATE}
    RECURRENT_DROPOUT: {RECURRENT_DROPOUT}
    EPOCHS: {EPOCHS}
    BATCH_SIZE: {BATCH_SIZE}

    {str(result)}

    """
    return { 
        "message" : message
    }

def execute_mining_steps(NUM_SAMPLES, MAX_LEN, TEST_SIZE, VOCAB_LENGTH, EMBEDDING_DIM, INPUT_LENGTH, DROP_RATE, RECURRENT_DROPOUT, EPOCHS, BATCH_SIZE):
    text_column = "description"
    class_column = "class"

    selector = Selector(text_column=text_column, class_column=class_column)
    df = selector.read_csv(path="data/news.psv")

    preprocessor = Preprocessor(text_column=text_column, class_column=class_column)
    df = preprocessor.remove_duplicated(df=df)
    df = preprocessor.remove_outliers(df=df)

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
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    evaluator = Evaluator(X_test=X_test, y_test=y_test)
    y_pred = evaluator.predict(model=model, batch_size=BATCH_SIZE)
    df_cm = evaluator.generate_confusion_matrix(y_pred=y_pred, labels=labels)
    report = evaluator.generate_report(labels=labels, df_cm=df_cm)

    return report
