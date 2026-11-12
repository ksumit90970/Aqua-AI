import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "water_model.pkl"
DB_PATH = ROOT / "database" / "water_data.db"

st.set_page_config(page_title="Smart Water Quality Monitor", page_icon="💧", layout="wide")

FEATURE_LABELS = {
    "ph": ("pH", 0.0, 14.0, 7.0, 0.1),
    "Hardness": ("Hardness", 0.0, 5000.0, 150.0, 1.0),
    "Solids": ("Total dissolved solids (Solids)", 0.0, 100000.0, 20000.0, 100.0),
    "Chloramines": ("Chloramines", 0.0, 50.0, 7.0, 0.1),
    "Sulfate": ("Sulfate", 0.0, 2000.0, 300.0, 1.0),
    "Conductivity": ("Conductivity", 0.0, 5000.0, 400.0, 1.0),
    "Organic_carbon": ("Organic carbon", 0.0, 100.0, 10.0, 0.1),
    "Trihalomethanes": ("Trihalomethanes", 0.0, 500.0, 60.0, 0.1),
    "Turbidity": ("Turbidity", 0.0, 100.0, 4.0, 0.1),
}

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ph REAL, Hardness REAL, Solids REAL, Chloramines REAL,
                Sulfate REAL, Conductivity REAL, Organic_carbon REAL,
                Trihalomethanes REAL, Turbidity REAL,
                prediction INTEGER, model_note TEXT
            )
        """)

def save_reading(values, prediction):
    cols = list(FEATURE_LABELS.keys())
    row = [datetime.now().isoformat(timespec="seconds")]
    row += [float(values[c]) for c in cols]
    row += [int(prediction), "Model output; not a laboratory test"]
    placeholders = ",".join(["?"] * len(row))
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            f"INSERT INTO readings (created_at,{','.join(cols)},prediction,model_note) VALUES ({placeholders})",
            row
        )

def load_history():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT * FROM readings ORDER BY id DESC", con)

init_db()

st.title("💧 Smart Water Quality Monitoring & Contamination Prediction")
st.caption("Software prototype • Dataset-based machine learning • No physical sensors required")

page = st.sidebar.radio("Navigate", ["Dashboard", "Predict water quality", "History", "About"])

if page == "Dashboard":
    st.subheader("Project dashboard")
    c1, c2, c3 = st.columns(3)
    history = load_history()
    c1.metric("Saved readings", len(history))
    c2.metric("Model available", "Yes" if MODEL_PATH.exists() else "No")
    c3.metric("Latest result", ("Potable class" if int(history.iloc[0]["prediction"]) == 1 else "Non-potable class") if not history.empty else "No readings")
    st.markdown("""
    This prototype accepts water-parameter values, applies a trained machine-learning model,
    displays its predicted class, and stores each result in a local SQLite database.
    """)
    if not history.empty:
        counts = history["prediction"].map({0: "Non-potable class (0)", 1: "Potable class (1)"}).value_counts().reset_index()
        counts.columns = ["Predicted class", "Readings"]
        st.plotly_chart(px.pie(counts, names="Predicted class", values="Readings", title="Saved prediction classes"), use_container_width=True)
    else:
        st.info("No saved predictions yet. Open 'Predict water quality' to try the prototype.")
    st.warning("A model prediction is not a drinking-water safety certificate. Confirm water quality with an accredited laboratory.")

elif page == "Predict water quality":
    st.subheader("Enter water parameters")
    if not MODEL_PATH.exists():
        st.error("The trained model was not found. First run: python train_model.py")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    st.write("Enter the measured values below. The default values are only examples.")
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        label, minimum, maximum, default, step = FEATURE_LABELS.get(
            feature, (feature, 0.0, 100000.0, 1.0, 0.1)
        )
        with cols[i % 3]:
            values[feature] = st.number_input(
                label, min_value=float(minimum), max_value=float(maximum),
                value=float(default), step=float(step), key=f"input_{feature}"
            )

    if st.button("Predict water quality", type="primary"):
        sample = pd.DataFrame([[values[f] for f in features]], columns=features)
        prediction = int(model.predict(sample)[0])
        save_reading(values, prediction)
        if prediction == 1:
            st.success("Model result: Potable class (1)")
        else:
            st.error("Model result: Non-potable class (0)")
        st.caption("This is a machine-learning classification based on the training dataset, not proof that the water is safe or unsafe.")
        st.info("For real use, compare measured values with applicable drinking-water standards and obtain laboratory confirmation.")

elif page == "History":
    st.subheader("Prediction history")
    history = load_history()
    if history.empty:
        st.info("No readings saved yet.")
    else:
        display = history.copy()
        display["prediction"] = display["prediction"].map({0: "Non-potable class (0)", 1: "Potable class (1)"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="water_prediction_history.csv",
            mime="text/csv"
        )

elif page == "About":
    st.subheader("About this prototype")
    st.markdown("""
    **Workflow**
    1. Load a labelled water-quality CSV dataset.
    2. Fill missing numeric values using median imputation.
    3. Train a Random Forest classifier.
    4. Evaluate it on a held-out test set.
    5. Use the saved model for new parameter inputs.
    6. Save each prediction in SQLite.

    **Dataset columns expected**
    `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`,
    `Organic_carbon`, `Trihalomethanes`, `Turbidity`, `Potability`.

    `Potability` must contain 0/1 labels. The included sample data is synthetic and is
    provided only to test that the software runs. Replace it with a real, appropriately
    licensed dataset before reporting model performance as meaningful.
    """)
    st.warning("The project does not identify specific contaminants. It predicts the dataset's Potability label only.")
