import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.impute import SimpleImputer

def load_datasets(trend_path="output/trend_dataset.csv", sent_path="output/reddit_comment_sentiment.csv"):
    trend_df = pd.read_csv(trend_path, dtype=str)
    sent_df = pd.read_csv(sent_path, dtype=str)

    # --- Ensure a common join key 'permalink' exists in both dataframes ---
    # If trend_df has 'permalink' already, great. If it has 'url', copy it.
    if "permalink" not in trend_df.columns:
        if "url" in trend_df.columns:
            trend_df["permalink"] = trend_df["url"]
        else:
            # try other fallback names
            for candidate in ("link","post_url","permalink_url"):
                if candidate in trend_df.columns:
                    trend_df["permalink"] = trend_df[candidate]
                    break

    # If sent_df uses a different name, normalize it too (rare)
    if "permalink" not in sent_df.columns:
        if "url" in sent_df.columns:
            sent_df["permalink"] = sent_df["url"]

    # Now merge on permalink safely
    df = pd.merge(trend_df, sent_df, on="permalink", how="left")
    return df


def prepare_features(df):
    df = df.copy()

    # Target variable
    y = df["engagement_label"]

    # Numeric features
    numeric = df[["score", "num_comments", "engagement_score",
                  "avg_neg", "avg_neu", "avg_pos", "avg_compound"]]

    # Categorical: subreddit, trend keywords
    cat_features = df[["subreddit"]].astype(str)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = enc.fit_transform(cat_features)

    X = np.hstack([numeric.values, cat_encoded])

    return X, y, enc

def train_models():
    df = load_datasets()
    X, y, encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Ensure X arrays are numeric numpy arrays (if X is a DataFrame this will work too)
    # This helps SimpleImputer behave predictably.
    import numpy as np
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    preds_rf = rf.predict(X_test)
    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, preds_rf))
    print(classification_report(y_test, preds_rf))

    # ----------------------------
    # Logistic Regression (with NaN handling via imputation)
    # ----------------------------
    print("\nTraining Logistic Regression...")

    # Impute missing values (mean strategy)
    imputer = SimpleImputer(strategy="mean")
    X_train_lr = imputer.fit_transform(X_train)
    X_test_lr = imputer.transform(X_test)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_lr, y_train)

    preds_lr = lr.predict(X_test_lr)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, preds_lr))
    print(classification_report(y_test, preds_lr))

    # Save best model (Random Forest) and also save imputer + encoder so pipeline can be reproduced
    with open("output/model.pkl", "wb") as f:
        pickle.dump({"model": rf, "encoder": encoder, "imputer": imputer}, f)

    print("\nSaved model to output/model.pkl")


if __name__ == "__main__":
    train_models()
