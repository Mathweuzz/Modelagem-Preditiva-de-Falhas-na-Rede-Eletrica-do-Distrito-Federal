import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress visual display in backend execution
import matplotlib
matplotlib.use('Agg')

def run_exploratory_data_analysis(dataset_path, output_dir_img='img'):
    """
    Main Exploration Pipeline running exhaustive Temporal,
    Seasonality, and Distribution diagnostics.
    """
    print(f"Loading massive climatological and outage tensor from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Unified Interruption Dataset missing at: {dataset_path}")
        
    df = pd.read_csv(dataset_path, parse_dates=['data'])
    df.set_index('data', inplace=True)
    
    # Ensure Output Directory exists
    os.makedirs(output_dir_img, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Pearson Correlation Heatmap (Inter-Variable Dynamics)
    # ---------------------------------------------------------
    print("Generating Pearson Correlation Triangular Heatmap...")
    plt.figure(figsize=(12, 10))
    # Select continuous numerical columns only
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr(method='pearson')
    
    # Mask the upper triangle for academic cleanliness
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title("Matriz de Correlacao de Pearson: Climatologia vs Interrupcoes", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_img, 'eda_heatmap_pearson.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Scatter Plot: Thermal Constraints vs Grid Resilience
    # ---------------------------------------------------------
    print("Generating Extreme Wind Scatter Plot...")
    if 'vento_rajada_max' in df.columns and 'interrupcoes' in df.columns:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=df, 
            x='vento_rajada_max', 
            y='interrupcoes', 
            hue='precipitacao_total',
            palette='viridis',
            alpha=0.7,
            edgecolor=None
        )
        plt.title('Dispersao Vetorial: Rajadas de Vento Extremos vs Quedas de Rede')
        plt.xlabel('Velocidade Maxima da Rajada (m/s)')
        plt.ylabel('Volume Diario de Interrupcoes (Falhas)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_img, 'eda_scatter_ventos.png'), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # 3. Monthly Seasonality (Boxplots / Violin)
    # ---------------------------------------------------------
    print("Generating Seasonal Outage Boxplots...")
    df['Mes'] = df.index.month
    
    plt.figure(figsize=(11, 6))
    sns.boxplot(x='Mes', y='interrupcoes', data=df, palette='Set3', fliersize=3)
    plt.title('Sazonalidade Mensal: Distribuicao de Interrupcoes no Cerrado (2016-2024)')
    plt.xlabel('Mes do Ano (1 = Janeiro, 12 = Dezembro)')
    plt.ylabel('Contagem Diaria de Ocorrencias')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_img, 'eda_boxplot_sazonalidade.png'), dpi=300)
    plt.close()

    print("Exploratory Data Analysis Pipeline Extracted Successfully.")

if __name__ == "__main__":
    # Mock entry point if executed identically
    # run_exploratory_data_analysis("../dados/processados/dataframe_final.csv")
    pass
