import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df = df.dropna()

    df["result"] = df["result"].map({"Fail": 0, "Pass": 1})

    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
