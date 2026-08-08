"""
04 - EDA Básica: Distribuição, Evolução Anual e Comparações Multi-Escala
=========================================================================

Gera os gráficos descritivos de alto nível usados no Cap. 4 da monografia:

  - serie_temporal_completa.png        (série diária + SMA-30 + corte treino/teste)
  - distribuicao_interrupcoes.png      (histograma + KDE com média e mediana)
  - evolucao_anual_interrupcoes.png    (média ± desvio-padrão por ano)
  - eda_violin_anomalias.png           (violinplot termodinâmica vs severidade da rede)

Fonte de dados:
  ../data/base_diaria_interrupcoes_clima_vento.csv

Saída:
  ../results/eda/

Execução:
  cd Fonte/src && python 04_eda_basica.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10
})

# Mesmo split usado pelos modelos: últimos 365 dias = teste
TEST_SIZE_DAYS = 365


def load_base(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    return df.sort_index()


def plot_serie_temporal_completa(df, save_path):
    """Série diária com SMA-30 e corte treino/teste."""
    print("Gerando série temporal completa...")
    serie = df['interrupcoes']
    sma30 = serie.rolling(30, min_periods=1).mean()
    split_date = serie.index[-TEST_SIZE_DAYS]

    plt.figure(figsize=(14, 5))
    plt.plot(serie.index, serie.values, color='gray', alpha=0.45,
             linewidth=0.7, label='Interrupções diárias')
    plt.plot(sma30.index, sma30.values, color='tab:red', linewidth=1.8,
             label='Média móvel (SMA-30)')
    plt.axvline(split_date, color='black', linestyle='--', alpha=0.7,
                label=f'Corte treino/teste ({split_date.date()})')

    plt.title('Série Temporal Completa de Interrupções Diárias no DF (2017–2025)')
    plt.xlabel('Data')
    plt.ylabel('Quantidade de Interrupções')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


def plot_distribuicao_interrupcoes(df, save_path):
    """Histograma + KDE com média (preta) e mediana (verde)."""
    print("Gerando distribuição das interrupções...")
    serie = df['interrupcoes'].dropna()
    media = serie.mean()
    mediana = serie.median()

    plt.figure(figsize=(12, 6))
    sns.histplot(serie, bins=60, kde=True, color='tab:blue',
                 edgecolor='white', alpha=0.7, stat='count')
    plt.axvline(media, color='black', linestyle='--', linewidth=2,
                label=f'Média = {media:.1f}')
    plt.axvline(mediana, color='tab:green', linestyle='-.', linewidth=2,
                label=f'Mediana = {mediana:.1f}')
    plt.title('Distribuição das Interrupções Diárias')
    plt.xlabel('Interrupções por dia')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


def plot_evolucao_anual(df, save_path):
    """Média ± desvio-padrão de interrupções diárias por ano.
       Anos puramente de treino: barra azul. Ano com dados de teste: vermelho."""
    print("Gerando evolução anual...")
    df_year = df.copy()
    df_year['ano'] = df_year.index.year
    grouped = df_year.groupby('ano')['interrupcoes'].agg(['mean', 'std', 'count'])

    split_date = df.index[-TEST_SIZE_DAYS]
    test_year = split_date.year
    cores = ['tab:blue' if ano < test_year else 'tab:red' for ano in grouped.index]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   color=cores, edgecolor='black', linewidth=0.7,
                   capsize=5, alpha=0.85)

    # Labels em cima das barras (igual ao TCC original)
    for bar, mean_val, std_val in zip(bars, grouped['mean'], grouped['std']):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 mean_val + std_val + 6,
                 f'{int(round(mean_val))}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color='tab:blue', alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color='tab:red', alpha=0.85)]
    plt.legend(handles, ['Treinamento', 'Inclui período de teste'], loc='upper left')

    plt.title('Evolução Anual das Interrupções — Neoenergia Brasília')
    plt.xlabel('Ano')
    plt.ylabel('Interrupções Diárias (Média ± Desvio)')
    plt.xticks(grouped.index)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


def plot_violin_anomalias(df, save_path):
    """Distribuição da temperatura por estrato de severidade da rede."""
    print("Gerando violin de anomalias térmicas vs severidade...")
    df2 = df[['interrupcoes', 'temperatura_media']].dropna().copy()
    df2['Severidade'] = pd.cut(
        df2['interrupcoes'],
        bins=[float('-inf'), 199, 400, float('inf')],
        labels=['Normal (<200)', 'Moderada (200--400)', 'Severa (>400)'],
        include_lowest=True,
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Severidade', y='temperatura_media', data=df2,
                   palette='husl', inner='quartile')
    plt.title('Distribuição Térmica por Faixa Canônica de Severidade')
    plt.xlabel('Faixa do alvo diário')
    plt.ylabel('Temperatura Média Diária (°C)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


if __name__ == "__main__":
    os.makedirs('../results/eda', exist_ok=True)

    data_path = '../data/base_diaria_interrupcoes_clima_vento.csv'
    df = load_base(data_path)

    plot_serie_temporal_completa(df, '../results/eda/serie_temporal_completa.png')
    plot_distribuicao_interrupcoes(df, '../results/eda/distribuicao_interrupcoes.png')
    plot_evolucao_anual(df, '../results/eda/evolucao_anual_interrupcoes.png')
    # Correlacoes multi-escala sao geradas apenas por 05_correlacoes_unificadas.py.
    plot_violin_anomalias(df, '../results/eda/eda_violin_anomalias.png')

    print("\n[OK] EDA básica concluída.")
