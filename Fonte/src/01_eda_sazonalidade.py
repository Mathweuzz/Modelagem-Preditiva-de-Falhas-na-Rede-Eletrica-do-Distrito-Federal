"""
01 - Análise Exploratória: Sazonalidade, Decomposição e Autocorrelação
=======================================================================

Gera os gráficos de:
  - decomposicao_interrupcoes.png  (decomposição aditiva STL: tendência, sazonalidade, resíduo)
  - autocorrelacao_interrupcoes.png (ACF e PACF da série de interrupções)
  - decomposicao_precipitacao.png  (decomposição da precipitação diária)

Fonte de dados:
  ../data/base_diaria_interrupcoes_clima_vento.csv
  (base diária consolidada INMET + ANEEL, 2017-2025)

Saída:
  ../results/eda/

Execução:
  cd Fonte/src && python 01_eda_sazonalidade.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


def load_data(filepath):
    print(f"Carregando dados de {filepath}...")
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    df = df.sort_index()
    return df


def plot_decomposition(df, column, title_suffix, save_path):
    print(f"Realizando decomposição sazonal aditiva de {column}...")
    series = df[column].interpolate(method='time')
    result = seasonal_decompose(series, model='additive', period=365)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    result.observed.plot(ax=axes[0], color='black')
    axes[0].set_ylabel('Observado')
    axes[0].set_title(f'Decomposição de Séries Temporais: {title_suffix}')

    result.trend.plot(ax=axes[1], color='tab:red')
    axes[1].set_ylabel('Tendência')

    result.seasonal.plot(ax=axes[2], color='tab:blue')
    axes[2].set_ylabel('Sazonalidade')

    result.resid.plot(ax=axes[3], color='tab:green', style='o', markersize=2)
    axes[3].set_ylabel('Resíduos')
    axes[3].set_xlabel('data')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


def plot_autocorrelation(df, column, title_suffix, save_path):
    print(f"Gerando ACF/PACF de {column}...")
    series = df[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_acf(series, ax=axes[0], lags=60, alpha=0.05, color='tab:blue')
    axes[0].set_title(f'Função de Autocorrelação (ACF) - {title_suffix}')
    axes[0].set_xlabel('Defasagem (Dias)')
    axes[0].set_ylabel('Autocorrelação')

    plot_pacf(series, ax=axes[1], lags=60, alpha=0.05, color='tab:orange')
    axes[1].set_title(f'Função de Autocorrelação Parcial (PACF) - {title_suffix}')
    axes[1].set_xlabel('Defasagem (Dias)')
    axes[1].set_ylabel('Autocorrelação Parcial')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


if __name__ == "__main__":
    os.makedirs('../results/eda', exist_ok=True)

    data_path = '../data/base_diaria_interrupcoes_clima_vento.csv'
    df = load_data(data_path)

    plot_decomposition(
        df=df,
        column='interrupcoes',
        title_suffix='Quantidade de Interrupções Diárias',
        save_path='../results/eda/decomposicao_interrupcoes.png'
    )

    plot_autocorrelation(
        df=df,
        column='interrupcoes',
        title_suffix='Interrupções',
        save_path='../results/eda/autocorrelacao_interrupcoes.png'
    )

    plot_decomposition(
        df=df,
        column='precipitacao_total_mm',
        title_suffix='Precipitação Diária',
        save_path='../results/eda/decomposicao_precipitacao.png'
    )

    print("\n[OK] Análise de Sazonalidade concluída.")
