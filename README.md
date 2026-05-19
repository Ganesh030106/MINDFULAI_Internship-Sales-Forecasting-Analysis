# MINDFULAI Internship — Sales Forecasting & Analysis

A complete ML pipeline for sales data preprocessing, anomaly detection, classification, regression, and time-series forecasting (Prophet) — with an interactive **Streamlit dashboard** for visual exploration.

---

## 🗂 Repository Structure

```
├── app/
│   └── dsboard.py              # Streamlit dashboard (main entry point)
├── data/
│   └── train.csv               # Raw Superstore sales dataset
├── reports/                    # Auto-generated plots & anomaly CSV
├── .streamlit/
│   └── config.toml             # Streamlit Cloud theme config
├── Cleaned_dataset.csv         # Preprocessed dataset (output of main.py)
├── Cleaned_dataset_no_anomalies.csv  # Dataset with anomalies removed
├── main.py                     # ML pipeline script (classification, regression, forecasting)
├── requirements.txt            # Python dependencies
└── README.md
```

## ✨ Features

- **Data Preprocessing** — Cleans, encodes, and engineers features from raw sales data
- **Anomaly Detection** — IQR-based outlier flagging with boxplot visualisation
- **Classification** — XGBoost with GridSearchCV to predict high-sales transactions
- **Regression** — Gradient Boosting to predict sales amounts
- **Time-Series Forecasting** — Facebook Prophet for 30-day sales forecast
- **Interactive Dashboard** — Streamlit app with KPI cards, charts, filters, and forecasting

## 📊 Dashboard Preview

The Streamlit dashboard (`app/dsboard.py`) includes:
- KPI cards (Total Sales, Order Count, Average Sale, Max Sale)
- Sales over time (line chart + Prophet forecast)
- Monthly sales trend
- Category & Region breakdowns
- Top 10 products
- Sunburst chart (Category → Sub-Category)
- Customer Lifetime Value distribution
- Ship Mode distribution & Sub-Category boxplots
- Cumulative sales growth

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (3.10 / 3.11 recommended)
- `pip`

### Install Dependencies
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If installing `prophet` on Windows fails, follow the [official Prophet install guide](https://facebook.github.io/prophet/docs/installation.html) or use WSL / a Linux environment.

### Run the ML Pipeline
```bash
python main.py
```
This will:
1. Load `data/train.csv`
2. Preprocess & detect anomalies
3. Train classification & regression models
4. Generate Prophet forecast
5. Save plots to `reports/` and cleaned CSVs to the project root

### Run the Dashboard Locally
```bash
streamlit run app/dsboard.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** and select:
   - **Repository:** `Ganesh030106/MINDFULAI_Internship-Sales-Forecasting-Analysis`
   - **Branch:** `main`
   - **Main file path:** `app/dsboard.py`
5. Click **Deploy** 🎉

---

## 📂 Outputs

| Output | Description |
|--------|-------------|
| `reports/*.png` | Diagnostic plots (boxplot, scatter, confusion matrix, ROC, regression, forecast) |
| `reports/anomaly_records.csv` | Flagged anomalous sales records |
| `Cleaned_dataset.csv` | Full preprocessed dataset |
| `Cleaned_dataset_no_anomalies.csv` | Preprocessed dataset with outliers removed |

---

## 🛠 Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, Prophet
- **Visualisation** — matplotlib, seaborn, Plotly
- **Dashboard** — Streamlit
- **Deployment** — Streamlit Community Cloud

---

## 📬 Contact

Built as part of the MINDFULAI Internship programme.