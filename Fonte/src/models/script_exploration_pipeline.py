"""
Pipeline Exploratório (EDA Inicial)
====================================

Gera os 3 gráficos de inspeção rápida usados para diagnóstico do dataset:

  - eda_heatmap_pearson.png       (matriz de correlação de Pearson)
  - eda_scatter_ventos.png        (rajada máxima vs interrupções, colorido por chuva)
  - eda_boxplot_sazonalidade.png  (distribuição mensal de interrupções)

Fonte de dados:
  ../../data/base_diaria_interrupcoes_clima_vento.csv

Saída padrão:
  ../../results/eda/

Execução:
  cd Fonte/src/models && python script_exploration_pipeline.py
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')


def run_exploratory_data_analysis(dataset_path, output_dir_img):
    print(f"Carregando base diária de: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset ausente: {dataset_path}")

    df = pd.read_csv(dataset_path, parse_dates=['data'])
    df.set_index('data', inplace=True)

    os.makedirs(output_dir_img, exist_ok=True)

    # 1. Matriz de Correlação de Pearson
    print("Gerando heatmap de Pearson...")
    plt.figure(figsize=(12, 10))
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0,
                center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .75})
    plt.title("Matriz de Correlação de Pearson: Climatologia vs Interrupções",
              fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_img, 'eda_heatmap_pearson.png'), dpi=300)
    plt.close()

    # 2. Scatter Rajada Máxima vs Interrupções (colorido por chuva)
    if 'vento_rajada_max_ms' in df.columns and 'interrupcoes' in df.columns:
        print("Gerando scatter rajada vs interrupções...")
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=df,
            x='vento_rajada_max_ms',
            y='interrupcoes',
            hue='precipitacao_total_mm',
            palette='viridis',
            alpha=0.7,
            edgecolor=None
        )
        plt.title('Dispersão: Rajadas de Vento Extremas vs Quedas de Rede')
        plt.xlabel('Velocidade Máxima da Rajada (m/s)')
        plt.ylabel('Volume Diário de Interrupções')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_img, 'eda_scatter_ventos.png'), dpi=300)
        plt.close()

    # 3. Sazonalidade Mensal (boxplot por mês)
    print("Gerando boxplot mensal...")
    df['Mes'] = df.index.month
    plt.figure(figsize=(11, 6))
    sns.boxplot(x='Mes', y='interrupcoes', data=df, palette='Set3', fliersize=3)
    plt.title('Sazonalidade Mensal: Distribuição de Interrupções no Cerrado (2017–2025)')
    plt.xlabel('Mês do Ano (1 = Janeiro, 12 = Dezembro)')
    plt.ylabel('Contagem Diária de Ocorrências')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_img, 'eda_boxplot_sazonalidade.png'), dpi=300)
    plt.close()

    print("[OK] EDA exploratória concluída.")


if __name__ == "__main__":
    run_exploratory_data_analysis(
        dataset_path='../../data/base_diaria_interrupcoes_clima_vento.csv',
        output_dir_img='../../results/eda'
    )
