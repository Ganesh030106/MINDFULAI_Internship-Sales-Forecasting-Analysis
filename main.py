import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

REPORT_FOLDER = "reports"

def ensure_reports_dir():
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)

def load_fresh_data(filepath):
    
    print("Loading data from:", filepath)
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("[INFO] Preprocessing Dataset")
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df.drop('Ship Date', axis=1)
    df = df.dropna(subset=['Order Date'])
    df['Postal Code'] = df['Postal Code'].fillna(-1)
    df = df.drop_duplicates()


    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Quarter'] = df['Order Date'].dt.quarter
    df['Order_Weekday'] = df['Order Date'].dt.weekday

    cat_cols = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df['High_Sales'] = (df['Sales'] > df['Sales'].median()).astype(int)
    return df, cat_cols


def detect_anomalies(df):
    #print("\n🚨 Anomaly Detection Report")
    q1 = df['Sales'].quantile(0.25)
    q3 = df['Sales'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomalies = df[(df['Sales'] < lower) | (df['Sales'] > upper)]
    print(f"Found {len(anomalies)} anomalous records.")
    anomalies.to_csv('reports/anomaly_records.csv', index=False)


    # Save anomaly plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Sales"])
    plt.title("Sales Anomaly Detection (Boxplot)")
    plt.xlabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_FOLDER, "1. sales_anomalies_boxplot.png"))

    plt.close()

    df_cleaned = df[~((df['Sales'] < lower) | (df['Sales'] > upper))].copy()
    print(f"Removed {len(anomalies)} anomalies. Remaining records: {len(df_cleaned)}")
    return df_cleaned


def plot_scatter(df, x_col, y_col, hue_col, title, filename):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[x_col], y=df[y_col], hue=df[hue_col], palette="coolwarm")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    path = os.path.join(REPORT_FOLDER, filename)
    plt.savefig(path)

    plt.close()
    print(f"Scatterplot saved: {path}")
    """
    XGBoost Classification with hyperparameter tuning, saved plots.
    """

def run_classification(df, features, target='High_Sales'):
    
    print("\n🔍 Classification: High Sales Prediction")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    param_grid = {
        'max_depth': [3, 5],
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
    }
    best_model = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_grid, cv=3
    )
    best_model.fit(X_train, y_train)
    print("Best Params:", best_model.best_params_)

    y_pred = best_model.predict(X_test)
    print(f"✅ Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_FOLDER, "4. classification_confusion_matrix.png"))

    plt.close()


    y_probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_FOLDER, "5. classification_roc_curve.png"))

    plt.close()


def run_regression(df, features, target='Sales'):
   

    print("\n🔧 Regression: Sales Prediction")
    reg_df = df[features + [target]].dropna()
    X = reg_df[features]
    y = reg_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")


    plt.figure(figsize=(6,5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_FOLDER, "6. regression_actual_vs_pred.png"))

    plt.close()


def run_forecasting(df):
    print("\n📈 Forecasting with Prophet")
    df_ts = df.groupby('Order Date')['Sales'].sum().reset_index().dropna()
    df_ts.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_ts)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    plt.title("Prophet Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    fig1.savefig(os.path.join(REPORT_FOLDER, "7. forecast_prophet.png"))

    plt.close(fig1)

    # Prophet components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    fig2.savefig(os.path.join(REPORT_FOLDER, "8. forecast_components.png"))

    plt.close(fig2)


def main():
    ensure_reports_dir()
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv")
    df = load_fresh_data(file_path)
    df, features = preprocess_data(df)

    plot_scatter(df, 'Category', 'Sales', 'High_Sales', 'Sales by Category', "2. scatter_sales_by_category.png")
    plot_scatter(df, 'Region', 'Sales', 'High_Sales', 'Sales by Region', "3. scatter_sales_by_region.png")

    detect_anomalies(df)

    ml_features = features + ['Order_Month', 'Order_Quarter', 'Order_Weekday']

    run_classification(df, ml_features)
    run_regression(df, ml_features)
    run_forecasting(df)
    df.to_csv("Cleaned_dataset.csv", index=False)

    df_no_anomalies = detect_anomalies(df.copy())

    cleaned_output_path = "Cleaned_dataset_no_anomalies.csv"
    df_no_anomalies.to_csv(cleaned_output_path, index=False)
    print(f"\nFinal cleaned dataset (without anomalies) saved to '{cleaned_output_path}'")

if __name__ == "__main__":
    main()
