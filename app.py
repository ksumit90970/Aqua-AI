from datetime import datetime
from pathlib import Path
import sqlite3
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "water_model.pkl"
DB_PATH = ROOT / "database" / "water_data.db"

st.set_page_config(
    page_title="Aqua-AI | Smart Water Quality Monitor",
    page_icon="💧",
    layout="wide",
)


def ensure_model():
  if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Training water quality ML model for first-time setup..."):
      subprocess.run(
          [sys.executable, str(ROOT / "train_model.py")],
          cwd=str(ROOT),
          check=True,
      )


# (Label, min, max, default, step, safe_min, safe_max)
FEATURE_SPECS = {
    "ph": ("pH Level", 0.0, 14.0, 7.2, 0.1, 6.5, 8.5),
    "Hardness": ("Hardness (mg/L)", 0.0, 5000.0, 150.0, 1.0, 60.0, 200.0),
    "Solids": (
        "Total Dissolved Solids (ppm)",
        0.0,
        100000.0,
        16000.0,
        100.0,
        0.0,
        20000.0,
    ),
    "Chloramines": ("Chloramines (ppm)", 0.0, 50.0, 6.8, 0.1, 0.0, 8.0),
    "Sulfate": ("Sulfate (mg/L)", 0.0, 2000.0, 290.0, 1.0, 0.0, 350.0),
    "Conductivity": (
        "Conductivity (μS/cm)",
        0.0,
        5000.0,
        380.0,
        1.0,
        0.0,
        450.0,
    ),
    "Organic_carbon": (
        "Organic Carbon (ppm)",
        0.0,
        100.0,
        10.0,
        0.1,
        0.0,
        12.0,
    ),
    "Trihalomethanes": (
        "Trihalomethanes (μg/L)",
        0.0,
        500.0,
        55.0,
        0.1,
        0.0,
        80.0,
    ),
    "Turbidity": ("Turbidity (NTU)", 0.0, 100.0, 2.8, 0.1, 0.0, 4.0),
}

HEALTHY_PRESET = {
    "ph": 7.2,
    "Hardness": 150.0,
    "Solids": 16000.0,
    "Chloramines": 6.8,
    "Sulfate": 290.0,
    "Conductivity": 380.0,
    "Organic_carbon": 10.0,
    "Trihalomethanes": 55.0,
    "Turbidity": 2.8,
}

UNHEALTHY_PRESET = {
    "ph": 3.5,
    "Hardness": 340.0,
    "Solids": 46000.0,
    "Chloramines": 12.5,
    "Sulfate": 480.0,
    "Conductivity": 720.0,
    "Organic_carbon": 24.0,
    "Trihalomethanes": 115.0,
    "Turbidity": 8.5,
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
  cols = list(FEATURE_SPECS.keys())
  row = [datetime.now().isoformat(timespec="seconds")]
  row += [float(values[c]) for c in cols]
  status_text = (
      "Healthy (Potable)" if prediction == 1 else "Unhealthy (Non-potable)"
  )
  row += [int(prediction), status_text]
  placeholders = ",".join(["?"] * len(row))
  with sqlite3.connect(DB_PATH) as con:
    con.execute(
        f"INSERT INTO readings (created_at,{','.join(cols)},prediction,model_note) VALUES ({placeholders})",
        row,
    )


def load_history():
  with sqlite3.connect(DB_PATH) as con:
    return pd.read_sql_query("SELECT * FROM readings ORDER BY id DESC", con)


ensure_model()
init_db()

st.title("💧 Aqua-AI: Smart Water Quality Monitoring & Health Analysis")
st.caption(
    "AI-powered water testing diagnostics • Healthy vs. Unhealthy"
    " classification & analytical reports"
)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard & Reports", "Test Water Quality", "History", "About"],
)

# ==================== 1. DASHBOARD & REPORTS ====================
if page == "Dashboard & Reports":
  st.subheader("Water Testing Overview & Analytical Reports")
  history = load_history()

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Total Water Tests", len(history))
  c2.metric("AI Model Status", "Active" if MODEL_PATH.exists() else "Missing")

  if not history.empty:
    latest_pred = int(history.iloc[0]["prediction"])
    healthy_count = int((history["prediction"] == 1).sum())
    unhealthy_count = int((history["prediction"] == 0).sum())
    c3.metric(
        "Latest Water Status",
        "HEALTHY 🟢" if latest_pred == 1 else "UNHEALTHY 🔴",
    )
    c4.metric(
        "Healthy vs Unhealthy", f"{healthy_count} Safe / {unhealthy_count} Unsafe"
    )
  else:
    c3.metric("Latest Water Status", "No tests yet")
    c4.metric("Healthy vs Unhealthy", "0 / 0")

  if not history.empty:
    history_asc = history.iloc[::-1].copy()
    history_asc["Test #"] = range(1, len(history_asc) + 1)
    history_asc["Health Status"] = history_asc["prediction"].map(
        {1: "Healthy (Potable)", 0: "Unhealthy (Non-potable)"}
    )

    col_left, col_right = st.columns(2)

    with col_left:
      counts = history_asc["Health Status"].value_counts().reset_index()
      counts.columns = ["Status", "Count"]
      fig_pie = px.pie(
          counts,
          names="Status",
          values="Count",
          title="Water Health Distribution",
          color="Status",
          color_discrete_map={
              "Healthy (Potable)": "#16a34a",
              "Unhealthy (Non-potable)": "#dc2626",
          },
          hole=0.4,
      )
      st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
      fig_trend = px.line(
          history_asc,
          x="Test #",
          y=["ph", "Turbidity", "Chloramines"],
          markers=True,
          title="Key Contamination Indicators Across Water Tests",
          labels={"value": "Measured Value", "variable": "Parameter"},
      )
      st.plotly_chart(fig_trend, use_container_width=True)

    # Bar chart of average values: Healthy vs Unhealthy
    avg_df = (
        history_asc.groupby("Health Status")[
            ["ph", "Turbidity", "Chloramines", "Organic_carbon"]
        ]
        .mean()
        .reset_index()
    )
    avg_melted = avg_df.melt(
        id_vars="Health Status", var_name="Parameter", value_name="Average"
    )
    fig_bar = px.bar(
        avg_melted,
        x="Parameter",
        y="Average",
        color="Health Status",
        barmode="group",
        title="Average Parameter Levels: Healthy vs. Unhealthy Samples",
        color_discrete_map={
            "Healthy (Potable)": "#16a34a",
            "Unhealthy (Non-potable)": "#dc2626",
        },
    )
    st.plotly_chart(fig_bar, use_container_width=True)
  else:
    st.info(
        "No water test reports yet. Go to **Test Water Quality** in the"
        " sidebar to run your first sample."
    )

# ==================== 2. TEST WATER QUALITY ====================
elif page == "Test Water Quality":
  st.subheader("Test Water Sample Health")
  if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

  bundle = joblib.load(MODEL_PATH)
  model = bundle["model"]
  features = bundle["features"]

  # Quick Demo Buttons
  b1, b2, _ = st.columns([1, 1, 2])
  if b1.button("Load Healthy Sample Preset"):
    for k, v in HEALTHY_PRESET.items():
      st.session_state[f"input_{k}"] = float(v)
  if b2.button("Load Unhealthy Sample Preset"):
    for k, v in UNHEALTHY_PRESET.items():
      st.session_state[f"input_{k}"] = float(v)

  values = {}
  cols = st.columns(3)
  for i, feature in enumerate(features):
    label, minimum, maximum, default, step, _, _ = FEATURE_SPECS.get(
        feature, (feature, 0.0, 100000.0, 1.0, 0.1, 0.0, 100.0)
    )
    if f"input_{feature}" not in st.session_state:
      st.session_state[f"input_{feature}"] = float(default)
    with cols[i % 3]:
      values[feature] = st.number_input(
          label,
          min_value=float(minimum),
          max_value=float(maximum),
          step=float(step),
          key=f"input_{feature}",
      )

  if st.button("Analyze Water Health", type="primary"):
    sample = pd.DataFrame([[values[f] for f in features]], columns=features)
    prediction = int(model.predict(sample)[0])

    # Get model probability confidence if supported
    confidence_text = ""
    if hasattr(model, "predict_proba"):
      probs = model.predict_proba(sample)[0]
      confidence = probs[prediction] * 100
      confidence_text = f" (AI Confidence: **{confidence:.1f}%**)"

    save_reading(values, prediction)

    st.markdown("---")
    if prediction == 1:
      st.success(
          f"### 🟢 WATER IS HEALTHY (Potable){confidence_text}\n"
          "This water sample meets safe drinking water patterns and is"
          " suitable for consumption."
      )
    else:
      st.error(
          f"### 🔴 WATER IS UNHEALTHY (Non-Potable / Contaminated){confidence_text}\n"
          "Warning: One or more chemical/physical parameters indicate"
          " contamination. Do NOT drink without filtration or treatment."
      )

    # Parameter-by-Parameter Diagnostic Table & Visual Report
    report_rows = []
    chart_labels, pct_of_limit, bar_colors = [], [], []
    for feat in features:
      label, _, _, _, _, safe_min, safe_max = FEATURE_SPECS[feat]
      val = values[feat]
      in_range = safe_min <= val <= safe_max
      report_rows.append({
          "Parameter": label,
          "Measured Value": val,
          "Safe Range": f"{safe_min} – {safe_max}",
          "Condition": "Healthy" if in_range else "Out of Range",
      })
      chart_labels.append(feat)
      pct_of_limit.append(round((val / safe_max) * 100, 1) if safe_max > 0 else 0)
      bar_colors.append("#16a34a" if in_range else "#dc2626")

    c_table, c_chart = st.columns([1.1, 1.2])
    with c_table:
      st.markdown("**Parameter Diagnostic Breakdown**")
      st.dataframe(
          pd.DataFrame(report_rows), use_container_width=True, hide_index=True
      )

    with c_chart:
      st.markdown("**Sample Level vs. Maximum Safe Threshold (%)**")
      fig_diag = go.Figure(
          go.Bar(
              x=pct_of_limit,
              y=chart_labels,
              orientation="h",
              marker_color=bar_colors,
              text=[f"{p}%" for p in pct_of_limit],
              textposition="auto",
          )
      )
      fig_diag.add_vline(
          x=100,
          line_dash="dash",
          line_color="orange",
          annotation_text="Safe Limit (100%)",
      )
      fig_diag.update_layout(
          xaxis_title="% of Safe Upper Limit (100% = Max Safe)",
          height=380,
          margin=dict(l=10, r=10, t=20, b=20),
      )
      st.plotly_chart(fig_diag, use_container_width=True)

# ==================== 3. HISTORY ====================
elif page == "History":
  st.subheader("Saved Water Testing Reports")
  history = load_history()
  if history.empty:
    st.info("No water tests recorded yet.")
  else:
    display = history.copy()
    display["Health Status"] = display["prediction"].map(
        {0: "Unhealthy (Non-potable)", 1: "Healthy (Potable)"}
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Full Report as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="water_health_report.csv",
        mime="text/csv",
    )

# ==================== 4. ABOUT ====================
elif page == "About":
  st.subheader("About Aqua-AI")
  st.markdown("""
    **Aqua-AI** evaluates 9 physicochemical water quality parameters using a Random Forest classifier
    to determine whether water is **Healthy (Potable)** or **Unhealthy (Non-potable)** and generates
    visual threshold reports for each sample.
    """)
  st.warning(
      "Disclaimer: This software prototype provides ML-based screening and does"
      " not replace certified laboratory testing."
  )