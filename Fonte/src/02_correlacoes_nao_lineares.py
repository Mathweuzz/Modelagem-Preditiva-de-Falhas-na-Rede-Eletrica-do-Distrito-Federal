"""
02 - Correlações Não-Lineares e Cross-Correlation
==================================================

Gera os gráficos de:
  - correlacao_spearman.png     (matriz de correlação de Spearman, rank-based)
  - correlacao_kendall.png      (matriz de correlação de Kendall, ordinal)
  - cross_corr_chuva_interrupcoes.png  (cross-correlation chuva(t-lag) vs interrupções(t))
  - cross_corr_vento_interrupcoes.png  (cross-correlation rajada(t-lag) vs interrupções(t))

Fonte de dados:
  ../data/base_diaria_interrupcoes_clima_vento.csv

Saída:
  ../results/eda/

Execução:
  cd Fonte/src && python 02_correlacoes_nao_lineares.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 12})


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    return df


def plot_correlation_matrix(df, method, title, save_path):
    print(f"Calculando matriz de correlação de {method.capitalize()}...")
    corr = df.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


def plot_cross_correlation(df, col_x, col_y, max_lag, title, save_path):
    print(f"Calculando cross-correlation entre {col_x} e {col_y}...")
    df_clean = df[[col_x, col_y]].dropna()
    x = df_clean[col_x].values
    y = df_clean[col_y].values

    # Normalização para [-1, 1]
    x = (x - np.mean(x)) / (np.std(x) * len(x))
    y = (y - np.mean(y)) / np.std(y)

    corr = np.correlate(y, x, mode='full')
    lags = np.arange(-len(x) + 1, len(x))

    lag_mask = (lags >= -max_lag) & (lags <= max_lag)
    filtered_lags = lags[lag_mask]
    filtered_corr = corr[lag_mask]

    plt.figure(figsize=(12, 5))
    try:
        plt.stem(filtered_lags, filtered_corr, basefmt=" ")
    except TypeError:
        plt.stem(filtered_lags, filtered_corr, basefmt=" ", use_line_collection=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.xlabel('Defasagem / Lag (Dias)')
    plt.ylabel('Correlação Cruzada')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


if __name__ == "__main__":
    os.makedirs('../results/eda', exist_ok=True)

    data_path = '../data/base_diaria_interrupcoes_clima_vento.csv'
    df = load_data(data_path)

    cols_of_interest = [
        'interrupcoes', 'temperatura_media', 'precipitacao_total_mm',
        'vento_velocidade_media_ms', 'vento_velocidade_max_ms', 'vento_rajada_max_ms'
    ]
    df_subset = df[cols_of_interest].copy()

    df_subset.columns = ['Interrupções', 'Temp. Média', 'Precipitação (mm)',
                         'Vento Médio', 'Vento Máximo', 'Rajada Máxima']

    plot_correlation_matrix(
        df_subset, method='spearman',
        title='Correlação de Spearman (Não-Linear) - Interrupções vs Clima',
        save_path='../results/eda/correlacao_spearman.png'
    )

    plot_correlation_matrix(
        df_subset, method='kendall',
        title='Correlação de Kendall (Ordinal) - Interrupções vs Clima',
        save_path='../results/eda/correlacao_kendall.png'
    )

    plot_cross_correlation(
        df, col_x='precipitacao_total_mm', col_y='interrupcoes', max_lag=14,
        title='Correlação Cruzada: Chuva (t-lag) influenciando Interrupções (t)',
        save_path='../results/eda/cross_corr_chuva_interrupcoes.png'
    )

    plot_cross_correlation(
        df, col_x='vento_rajada_max_ms', col_y='interrupcoes', max_lag=14,
        title='Correlação Cruzada: Rajadas de Vento (t-lag) influenciando Interrupções (t)',
        save_path='../results/eda/cross_corr_vento_interrupcoes.png'
    )

    print("\n[OK] Correlações não-lineares concluídas.")
