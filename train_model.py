from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "water_quality.csv"
SAMPLE_PATH = ROOT / "data" / "sample_water_quality.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "water_model.pkl"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"

def main():
    # Prefer the real dataset if the user has added it; otherwise use demo data.
    source = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
    print(f"Loading dataset: {source.relative_to(ROOT)}")
    data = pd.read_csv(source)

    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[required].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].isin([0, 1])]

    if data[TARGET].nunique() != 2:
        raise ValueError("Potability must contain both classes: 0 and 1.")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nEvaluation on held-out test data")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "training_source": source.name}, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH.relative_to(ROOT)}")
    if source == SAMPLE_PATH:
        print("NOTE: This model was trained on synthetic demo data. Replace it with real data for meaningful evaluation.")

if __name__ == "__main__":
    main()
