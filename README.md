# Smart Water Quality Monitoring & Contamination Prediction

A beginner-friendly, software-only college project prototype. It uses Python, Streamlit,
scikit-learn, Plotly, and SQLite. No Arduino, ESP32, or physical sensors are required.

## 1. What is included?

- `app.py` — Streamlit dashboard, prediction form, history, CSV export.
- `train_model.py` — trains and evaluates a Random Forest classification model.
- `data/sample_water_quality.csv` — synthetic demo data so you can run the prototype immediately.
- `data/water_quality.csv` — place your real labelled dataset here (not included).
- `model/` — trained model is saved here after training.
- `database/` — SQLite database is created automatically when the app starts.

## 2. Install Python and open the project

Install Python 3.10 or newer and VS Code. Open this project folder in VS Code.

Open **Terminal → New Terminal** and run:

### Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```cmd
venv\Scripts\activate.bat
```

Install the packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the model

Run:
```bash
python train_model.py
```

If `data/water_quality.csv` exists, the script uses it. Otherwise it uses the included
synthetic demo dataset. The trained model is saved as `model/water_model.pkl`.

## 4. Start the web app

Run:
```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at:
`http://localhost:8501`

## 5. Use a real dataset

Download a labelled water-potability CSV, for example the Kaggle Water Potability dataset:
https://www.kaggle.com/datasets/adityakadiwal/water-potability

Place the CSV in the `data` folder and rename it to:
`water_quality.csv`

The file should contain these columns (spelling and capitalization matter):
- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`
- `Potability`

Then retrain:
```bash
python train_model.py
```
Stop and restart Streamlit if it is already running.

## 6. Important project limitations

- The included CSV is **synthetic demo data**, not real water-testing evidence.
- The demo model is only for checking that the application workflow works.
- The Kaggle-style `Potability` target is a binary class (0/1). It does not identify a
  particular contaminant or provide a three-level risk rating.
- A model prediction is not a laboratory result or a drinking-water safety certificate.
  Real decisions require validated measurements, relevant local standards, and laboratory
  confirmation.
- This version does not forecast future contamination trends. Forecasting requires
  timestamped measurements collected repeatedly over time, ideally for the same locations.

## 7. Troubleshooting

**`python` is not recognized:** reinstall Python and select “Add Python to PATH”, then reopen VS Code.

**`No module named streamlit` (or another package):** activate the virtual environment and run:
`pip install -r requirements.txt`

**Model not found:** run `python train_model.py` before `streamlit run app.py`.

**Missing columns:** check the CSV column names against the list in section 5.

**Port already in use:** run `streamlit run app.py --server.port 8502`.
