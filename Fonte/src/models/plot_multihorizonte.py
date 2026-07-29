"""
Geração dos gráficos finais de previsão multi-horizonte.
Lê apenas o CSV de métricas — não precisa rodar os modelos novamente.

Execução:
  cd Fonte/src/models && python plot_multihorizonte.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SAVE_PATH    = '../../results/ml'
METRICAS_CSV = f'{SAVE_PATH}/previsao_multihorizonte_metricas.csv'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

CORES     = {'XGBoost': 'tab:orange', 'Bi-LSTM': 'tab:green', 'Bi-GRU': 'tab:red'}
MARCAS    = {'XGBoost': 'o',          'Bi-LSTM': 's',         'Bi-GRU': '^'}
HORIZONTES = [1, 3, 7, 14]


def grafico_degradacao_mae(df, save_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # offsets verticais manuais para evitar sobreposição nas anotações
    # (h, modelo) -> offset em pontos
    offsets_custom = {
        (1,  'XGBoost'): (+14, 'center'),
        (1,  'Bi-LSTM'): (-20, 'center'),
        (1,  'Bi-GRU'):  (-16, 'center'),
        (3,  'XGBoost'): (+16, 'center'),
        (3,  'Bi-LSTM'): (-20, 'center'),
        (3,  'Bi-GRU'):  (-18, 'center'),
        (7,  'XGBoost'): (+14, 'center'),
        (7,  'Bi-LSTM'): (-18, 'center'),
        (7,  'Bi-GRU'):  (-18, 'center'),
        (14, 'XGBoost'): (+14, 'center'),
        (14, 'Bi-LSTM'): (+12, 'center'),
        (14, 'Bi-GRU'):  (-20, 'center'),
    }

    for modelo in ['XGBoost', 'Bi-LSTM', 'Bi-GRU']:
        sub  = df[df['Modelo'] == modelo].sort_values('Horizonte')
        maes = sub['MAE'].values
        ax.plot(HORIZONTES, maes,
                marker=MARCAS[modelo], color=CORES[modelo],
                linewidth=2.5, markersize=9, label=modelo, zorder=3)
        for h, mae in zip(HORIZONTES, maes):
            oy, ha = offsets_custom[(h, modelo)]
            ax.annotate(f'{mae:.0f}',
                        xy=(h, mae), xytext=(0, oy),
                        textcoords='offset points',
                        ha=ha, fontsize=11,
                        color=CORES[modelo], fontweight='bold')

    ax.set_xticks(HORIZONTES)
    ax.set_xticklabels([f'{h} dia{"s" if h > 1 else ""}' for h in HORIZONTES])
    ax.set_xlabel('Horizonte de previsão')
    ax.set_ylabel('MAE (interrupções/dia)')
    ax.set_title('Desempenho por Horizonte de Previsão - MAE')
    ax.legend(framealpha=0.95, loc='upper right')
    ax.set_ylim(0, max(df['MAE']) * 1.30)
    ax.set_xlim(0.5, 14.5)
    plt.tight_layout()

    out = f'{save_path}/previsao_multihorizonte_degradacao.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f'Salvo: {out}')


def grafico_degradacao_rmse(df, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for modelo in ['XGBoost', 'Bi-LSTM', 'Bi-GRU']:
        sub   = df[df['Modelo'] == modelo].sort_values('Horizonte')
        rmses = sub['RMSE'].values
        ax.plot(HORIZONTES, rmses,
                marker=MARCAS[modelo], color=CORES[modelo],
                linewidth=2.2, markersize=8, label=modelo,
                linestyle='--', zorder=3)
        for h, r in zip(HORIZONTES, rmses):
            ax.annotate(f'{r:.0f}',
                        xy=(h, r),
                        xytext=(0, 9),
                        textcoords='offset points',
                        ha='center', fontsize=10,
                        color=CORES[modelo], fontweight='bold')

    ax.set_xticks(HORIZONTES)
    ax.set_xticklabels([f'{h} dia{"s" if h > 1 else ""}' for h in HORIZONTES])
    ax.set_xlabel('Horizonte de previsão')
    ax.set_ylabel('RMSE (interrupções/dia)')
    ax.set_title('Desempenho por Horizonte de Previsão - RMSE')
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(df['RMSE']) * 1.25)
    plt.tight_layout()

    out = f'{save_path}/previsao_multihorizonte_degradacao_rmse.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f'Salvo: {out}')


def grafico_mae_rmse_duplo(df, save_path):
    """Um único painel com MAE e RMSE lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metrica, titulo in zip(
        axes,
        ['MAE', 'RMSE'],
        ['MAE por Horizonte de Previsão', 'RMSE por Horizonte de Previsão']
    ):
        for modelo in ['XGBoost', 'Bi-LSTM', 'Bi-GRU']:
            sub  = df[df['Modelo'] == modelo].sort_values('Horizonte')
            vals = sub[metrica].values
            ax.plot(HORIZONTES, vals,
                    marker=MARCAS[modelo], color=CORES[modelo],
                    linewidth=2.2, markersize=8, label=modelo, zorder=3)
            for h, v in zip(HORIZONTES, vals):
                ax.annotate(f'{v:.0f}',
                            xy=(h, v), xytext=(0, 9),
                            textcoords='offset points',
                            ha='center', fontsize=9.5,
                            color=CORES[modelo], fontweight='bold')

        ax.set_xticks(HORIZONTES)
        ax.set_xticklabels([f'{h}d' for h in HORIZONTES])
        ax.set_xlabel('Horizonte de previsão (dias)')
        ax.set_ylabel(f'{metrica} (interrupções/dia)')
        ax.set_title(titulo)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_ylim(0, max(df[metrica]) * 1.28)
        ax.legend(framealpha=0.9)

    plt.suptitle('Desempenho dos Modelos por Horizonte de Previsão',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = f'{save_path}/previsao_multihorizonte_mae_rmse.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f'Salvo: {out}')


if __name__ == '__main__':
    os.makedirs(SAVE_PATH, exist_ok=True)
    df = pd.read_csv(METRICAS_CSV)

    grafico_degradacao_mae(df, SAVE_PATH)
    grafico_degradacao_rmse(df, SAVE_PATH)
    grafico_mae_rmse_duplo(df, SAVE_PATH)

    print('[OK] Gráficos gerados.')
