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
    plt.xlabel('Residual Error (Real - Predicted Outages)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kde_residuos_modelos.png'), dpi=300)
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


def consolidate_predictions(results_dir):
    """
    Lê os CSVs `predictions_xgboost.csv`, `predictions_lstm_bi.csv` e
    `predictions_gru_bi.csv` (gerados por evaluate_and_plot) e retorna
    um DataFrame único com colunas Real_Outages, Pred_XGB, Pred_LSTM, Pred_GRU.
    """
    xgb = pd.read_csv(os.path.join(results_dir, 'predictions_xgboost.csv'),
                      index_col='data', parse_dates=True)
    lstm = pd.read_csv(os.path.join(results_dir, 'predictions_lstm_bi.csv'),
                       index_col='data', parse_dates=True)
    gru = pd.read_csv(os.path.join(results_dir, 'predictions_gru_bi.csv'),
                      index_col='data', parse_dates=True)

    df = pd.DataFrame(index=xgb.index)
    df['Real_Outages'] = xgb['real']
    df['Pred_XGB'] = xgb['pred']
    # Todos os CSVs oficiais usam as mesmas 365 datas-alvo do conjunto de teste.
    df['Pred_LSTM'] = lstm['pred'].reindex(df.index)
    df['Pred_GRU'] = gru['pred'].reindex(df.index)
    df = df.dropna()
    return df


if __name__ == "__main__":
    results_dir = '../../results/ml'
    out_dir = '../../results/ml'
    os.makedirs(out_dir, exist_ok=True)

    df = consolidate_predictions(results_dir)
    print(f"Linhas consolidadas: {len(df)}")

    plot_residual_kde(df, output_dir=out_dir)
    plot_heteroscedasticity_scatter(df, output_dir=out_dir)
    print("[OK] Gráficos avançados gerados.")
