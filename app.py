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


# (Label, min_input, max_input, default, step, ideal_val, healthy_min, healthy_max, unhealthy_desc)
FEATURE_SPECS = {
    "ph": ("pH Level", 0.0, 14.0, 7.2, 0.1, 7.0, 6.5, 8.5, "< 6.5 or > 8.5"),
    "Hardness": (
        "Hardness (mg/L)",
        0.0,
        5000.0,
        150.0,
        1.0,
        120.0,
        60.0,
        200.0,
        "< 60 or > 200",
    ),
    "Solids": (
        "Total Dissolved Solids (ppm)",
        0.0,
        100000.0,
        16000.0,
        100.0,
        15000.0,
        5000.0,
        20000.0,
        "> 20,000",
    ),
    "Chloramines": (
        "Chloramines (ppm)",
        0.0,
        50.0,
        6.8,
        0.1,
        6.0,
        4.0,
        8.0,
        "< 4.0 or > 8.0",
    ),
    "Sulfate": (
        "Sulfate (mg/L)",
        0.0,
        2000.0,
        290.0,
        1.0,
        250.0,
        150.0,
        350.0,
        "> 350",
    ),
    "Conductivity": (
        "Conductivity (μS/cm)",
        0.0,
        5000.0,
        380.0,
        1.0,
        350.0,
        250.0,
        450.0,
        "< 250 or > 450",
    ),
    "Organic_carbon": (
        "Organic Carbon (ppm)",
        0.0,
        100.0,
        10.0,
        0.1,
        8.0,
        2.0,
        12.0,
        "> 12.0",
    ),
    "Trihalomethanes": (
        "Trihalomethanes (μg/L)",
        0.0,
        500.0,
        55.0,
        0.1,
        50.0,
        10.0,
        80.0,
        "> 80.0",
    ),
    "Turbidity": (
        "Turbidity (NTU)",
        0.0,
        100.0,
        2.8,
        0.1,
        2.0,
        0.5,
        4.0,
        "> 4.0",
    ),
}

HEALTHY_PRESET = {
    "ph": 7.1,
    "Hardness": 125.0,
    "Solids": 15200.0,
    "Chloramines": 6.2,
    "Sulfate": 255.0,
    "Conductivity": 355.0,
    "Organic_carbon": 8.2,
    "Trihalomethanes": 52.0,
    "Turbidity": 2.1,
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
    "AI-powered water testing diagnostics • Ideal range benchmarks, deviation"
    " tracking & health classification"
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
  st.subheader("Test Water Sample & Ideal Deviation Analysis")
  if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

  bundle = joblib.load(MODEL_PATH)
  model = bundle["model"]
  features = bundle["features"]

  # Reference guide expander so users can view Healthy vs Unhealthy ranges anytime
  with st.expander(
      "View Ideal Values, Healthy Ranges & Unhealthy Thresholds Reference",
      expanded=False,
  ):
    ref_rows = []
    for feat in features:
      label, _, _, _, _, ideal_v, h_min, h_max, unh_desc = FEATURE_SPECS[feat]
      ref_rows.append({
          "Parameter": label,
          "Ideal Target": ideal_v,
          "Healthy Range": f"{h_min} – {h_max}",
          "Unhealthy Range": unh_desc,
      })
    st.dataframe(
        pd.DataFrame(ref_rows), use_container_width=True, hide_index=True
    )

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
    label, minimum, maximum, default, step, ideal_v, h_min, h_max, _ = (
        FEATURE_SPECS.get(
            feature, (feature, 0.0, 100000.0, 1.0, 0.1, 1.0, 0.0, 100.0, "> 100")
        )
    )
    if f"input_{feature}" not in st.session_state:
      st.session_state[f"input_{feature}"] = float(default)
    with cols[i % 3]:
      values[feature] = st.number_input(
          f"{label}  (Ideal: {ideal_v} | Healthy: {h_min}–{h_max})",
          min_value=float(minimum),
          max_value=float(maximum),
          step=float(step),
          key=f"input_{feature}",
      )

  if st.button("Analyze Water Health & Deviations", type="primary"):
    sample = pd.DataFrame([[values[f] for f in features]], columns=features)
    prediction = int(model.predict(sample)[0])

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
          "This water sample aligns with safe drinking water benchmarks and is"
          " suitable for consumption."
      )
    else:
      st.error(
          f"### 🔴 WATER IS UNHEALTHY (Non-Potable / Contaminated){confidence_text}\n"
          "Warning: Significant deviations from ideal water standards detected."
          " Do NOT drink without treatment."
      )

    # Build Detailed Deviation & Range Table
    report_rows = []
    chart_labels, pct_deviations, bar_colors = [], [], []
    healthy_param_count = 0

    for feat in features:
      label, _, _, _, _, ideal_v, h_min, h_max, unh_desc = FEATURE_SPECS[feat]
      val = values[feat]

      # Numeric and percentage deviation from ideal value
      diff = val - ideal_v
      pct_dev = ((val - ideal_v) / ideal_v * 100) if ideal_v != 0 else 0.0

      in_healthy_range = h_min <= val <= h_max
      if in_healthy_range:
        healthy_param_count += 1
        if abs(pct_dev) <= 10.0:
          status_tag = "🟢 Optimal (Near Ideal)"
          color = "#16a34a"
        else:
          status_tag = "🟡 Healthy (Within Range)"
          color = "#eab308"
      else:
        status_tag = "🔴 Unhealthy (Out of Range)"
        color = "#dc2626"

      report_rows.append({
          "Parameter": label,
          "Measured": round(val, 2),
          "Ideal Value": ideal_v,
          "Deviation from Ideal": f"{diff:+.2f} ({pct_dev:+.1f}%)",
          "Healthy Range": f"{h_min} – {h_max}",
          "Unhealthy Range": unh_desc,
          "Status": status_tag,
      })

      chart_labels.append(feat)
      pct_deviations.append(round(pct_dev, 1))
      bar_colors.append(color)

    m1, m2, m3 = st.columns(3)
    m1.metric("Parameters in Healthy Range", f"{healthy_param_count} / {len(features)}")
    m2.metric("Parameters Out of Range", f"{len(features) - healthy_param_count} / {len(features)}")
    avg_abs_dev = sum(abs(d) for d in pct_deviations) / len(pct_deviations)
    m3.metric("Mean Deviation from Ideal", f"{avg_abs_dev:.1f}%")

    st.markdown("### Parameter Benchmark & Deviation Report")
    st.dataframe(
        pd.DataFrame(report_rows), use_container_width=True, hide_index=True
    )

    st.markdown("### Percentage Deviation from Ideal Target (`0%` = Exact Ideal)")
    fig_dev = go.Figure(
        go.Bar(
            x=pct_deviations,
            y=chart_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:+.1f}%" for p in pct_deviations],
            textposition="auto",
        )
    )
    fig_dev.add_vline(
        x=0,
        line_dash="solid",
        line_color="#38bdf8",
        annotation_text="Ideal Target (0%)",
    )
    fig_dev.update_layout(
        xaxis_title="% Deviation from Ideal Value (- Below Ideal | + Above Ideal)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=20),
    )
    st.plotly_chart(fig_dev, use_container_width=True)

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
        file_name="water_health_deviation_report.csv",
        mime="text/csv",
    )

# ==================== 4. ABOUT ====================
elif page == "About":
  st.subheader("About Aqua-AI")
  st.markdown("""
    **Aqua-AI** evaluates 9 physicochemical water quality parameters using a Random Forest classifier
    to determine whether water is **Healthy (Potable)** or **Unhealthy (Non-potable)**, while benchmarking
    each reading against ideal target values and safe/unhealthy ranges.
    """)
  st.warning(
      "Disclaimer: This software prototype provides ML-based screening and does"
      " not replace certified laboratory testing."
  )