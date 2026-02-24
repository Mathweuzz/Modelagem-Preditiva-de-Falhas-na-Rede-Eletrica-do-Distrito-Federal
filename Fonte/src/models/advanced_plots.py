import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_predictions(predictions_csv_path):
    """
    Loads the synchronized DataFrame containing True Targets and 
    the predictions from LSTM, GRU, and XGBoost.
    """
    if not os.path.exists(predictions_csv_path):
        raise FileNotFoundError(f"Model predictions file not found: {predictions_csv_path}")
    return pd.read_csv(predictions_csv_path, index_col='data', parse_dates=True)

def plot_residual_kde(df, output_dir='img'):
    """
    Calculates Residual Errors (Real - Pred) and plots their
    Kernel Density Estimation (KDE) to visually inspect Heteroscedasticity.
    """
    print("Generating Residual KDE distributions...")
    plt.figure(figsize=(10, 6))
    
    df['Residual_LSTM'] = df['Real_Outages'] - df['Pred_LSTM']
    df['Residual_GRU'] = df['Real_Outages'] - df['Pred_GRU']
    df['Residual_XGB'] = df['Real_Outages'] - df['Pred_XGB']
    
    sns.kdeplot(df['Residual_LSTM'], label='Bi-LSTM Residuals', fill=True, alpha=0.4)
    sns.kdeplot(df['Residual_GRU'], label='Bi-GRU Residuals', fill=True, alpha=0.4)
    sns.kdeplot(df['Residual_XGB'], label='XGBoost Residuals', fill=True, alpha=0.4)
    
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Kernel Density Estimation of Predictive Residuals (Test Set)')
    plt.xlabel('Absolute Error (Real - Predicted Outages)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_residuos_modelos.png'), dpi=300)
    plt.close()

def plot_zoomed_anomaly(df, start_date='2023-11-01', end_date='2023-12-15', output_dir='img'):
    """
    Zooms into a specific chronological interval known for severe
    thermodynamic anomalies (e.g., El Nino Late-2023 storms).
    """
    print(f"Generating Zoomed Timeseries for anomaly interval: {start_date} to {end_date}...")
    mask = (df.index >= start_date) & (df.index <= end_date)
    zoomed_df = df.loc[mask]
    
    plt.figure(figsize=(14, 6))
    plt.plot(zoomed_df.index, zoomed_df['Real_Outages'], label='Real Observations (ANEEL)', 
             color='black', marker='o', linewidth=2)
    plt.plot(zoomed_df.index, zoomed_df['Pred_LSTM'], label='Bi-LSTM Prediction', 
             linestyle='--', color='blue', linewidth=2)
    plt.plot(zoomed_df.index, zoomed_df['Pred_XGB'], label='XGBoost Prediction', 
             linestyle=':', color='red', linewidth=2)
    
    plt.title('Model Stress Test Snapshot: Severe Outage Anomalies (El Nino 2023)')
    plt.ylabel('Number of Transformer Failures')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zoom_serie_2023_anomalia.png'), dpi=300)
    plt.close()

def plot_heteroscedasticity_scatter(df, output_dir='img'):
    """
    Scatters the Absolute Error against the Real Volumetric scale 
    to visually demonstrate variance scaling (Heteroscedasticity).
    """
    print("Generating Heteroscedasticity Scatter plot...")
    plt.figure(figsize=(9, 6))
    abs_error_lstm = np.abs(df['Real_Outages'] - df['Pred_LSTM'])
    
    plt.scatter(df['Real_Outages'], abs_error_lstm, alpha=0.5, color='darkblue')
    
    # Fit a standard linear trendline to show the increasing wedge
    z = np.polyfit(df['Real_Outages'], abs_error_lstm, 1)
    p = np.poly1d(z)
    plt.plot(df['Real_Outages'], p(df['Real_Outages']), "r--", linewidth=2.5, label='Variance Expansion Trend')
    
    plt.title('Absolute Predictive Error vs Outage Volume Scale (Bi-LSTM)')
    plt.xlabel('Volume of Real Outages (Daily Count)')
    plt.ylabel('Absolute Prediction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_heteroscedasticity.png'), dpi=300)
    plt.close()
