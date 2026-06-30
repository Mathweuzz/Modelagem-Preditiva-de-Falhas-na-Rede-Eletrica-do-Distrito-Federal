"""
Previsão Multi-Horizonte — Degradação do MAE por Horizonte
===========================================================

Treina cada modelo uma vez (com os últimos MAX_HORIZONTE dias reservados)
e avalia em múltiplos horizontes: x = 1, 3, 7, 14 dias.

Abordagem:
  - XGBoost: previsão direta com as features pré-computadas
  - Bi-LSTM / Bi-GRU: previsão recursiva (auto-regressiva)
      janela inicial = últimos SEQ_LENGTH dias do treino (valores reais)
      para cada passo h: prediz, insere previsão na janela, desliza

Artefatos gerados em results/ml/:
  - previsao_multihorizonte_degradacao.png    (MAE por horizonte, todos os modelos)
  - previsao_multihorizonte_barras_x3.png     (real vs previsto, x=3 dias)
  - previsao_multihorizonte_metricas.csv      (MAE e RMSE por modelo e horizonte)

Execução:
  cd Fonte/src/models && python previsao_multihorizonte.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from data_loader_dl import create_sequences
from lstm_bidirecional import AdvancedLSTM, train_dl_model
from gru_avancada import AdvancedGRU

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 12})

# ── Configuração ─────────────────────────────────────────────────────────────
HORIZONTES    = [1, 3, 7, 14]
MAX_HORIZONTE = max(HORIZONTES)
SEQ_LENGTH    = 14
TARGET_COL    = 'interrupcoes'
DATA_PATH     = '../../data/dataset_engenharia_features.csv'
SAVE_PATH     = '../../results/ml'


# ── Dados ────────────────────────────────────────────────────────────────────

def carregar(filepath):
    df = pd.read_csv(filepath, index_col='data', parse_dates=True)
    split = len(df) - MAX_HORIZONTE
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    print(f"Treino: {len(train_df)} dias  ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"Teste:  {len(test_df)} dias   ({test_df.index[0].date()} → {test_df.index[-1].date()})")
    return train_df, test_df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ── XGBoost ──────────────────────────────────────────────────────────────────

def rodar_xgboost(train_df, test_df):
    print("\n── XGBoost ──────────────────────────────")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].values

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    y_pred = np.maximum(0, model.predict(X_test))

    metricas = {}
    for h in HORIZONTES:
        mae_h  = mean_absolute_error(y_test[:h], y_pred[:h])
        rmse_h = rmse(y_test[:h], y_pred[:h])
        metricas[h] = (mae_h, rmse_h)
        print(f"  x={h:2d}d → MAE: {mae_h:.2f}  RMSE: {rmse_h:.2f}")

    return y_pred, y_test, test_df.index, metricas


# ── DL recursivo ─────────────────────────────────────────────────────────────

def prever_recursivo(model, train_scaled, test_df, scaler, target_idx, device):
    n_features = train_scaled.shape[1]
    janela = train_scaled[-SEQ_LENGTH:].copy()
    y_test = test_df[TARGET_COL].values

    previsoes_norm = []
    for _ in range(MAX_HORIZONTE):
        x_t = torch.from_numpy(janela).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x_t).cpu().numpy().flatten()[0]
        previsoes_norm.append(pred_norm)

        novo_dia = janela[-1].copy()
        novo_dia[target_idx] = pred_norm
        janela = np.vstack([janela[1:], novo_dia])

    dummy = np.zeros((MAX_HORIZONTE, n_features))
    dummy[:, target_idx] = previsoes_norm
    y_pred = np.maximum(0, scaler.inverse_transform(dummy)[:, target_idx])

    metricas = {}
    for h in HORIZONTES:
        mae_h  = mean_absolute_error(y_test[:h], y_pred[:h])
        rmse_h = rmse(y_test[:h], y_pred[:h])
        metricas[h] = (mae_h, rmse_h)
        print(f"  x={h:2d}d → MAE: {mae_h:.2f}  RMSE: {rmse_h:.2f}")

    return y_pred, metricas


def rodar_lstm(train_df, test_df):
    print("\n── Bi-LSTM ──────────────────────────────")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    target_idx = train_df.columns.get_loc(TARGET_COL)

    X_tr, y_tr = create_sequences(train_scaled, target_idx, SEQ_LENGTH)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float().unsqueeze(1)

    input_dim = X_tr_t.shape[2]
    model = AdvancedLSTM(input_size=input_dim, hidden_size=64, num_layers=2,
                         output_size=1, dropout_rate=0.4)
    model, _ = train_dl_model(model, X_tr_t, y_tr_t, epochs=150, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return prever_recursivo(model, train_scaled, test_df, scaler, target_idx, device)


def rodar_gru(train_df, test_df):
    print("\n── Bi-GRU ───────────────────────────────")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    target_idx = train_df.columns.get_loc(TARGET_COL)

    X_tr, y_tr = create_sequences(train_scaled, target_idx, SEQ_LENGTH)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float().unsqueeze(1)

    input_dim = X_tr_t.shape[2]
    model = AdvancedGRU(input_size=input_dim, hidden_size=64, num_layers=2,
                        output_size=1, dropout_rate=0.4)
    model, _ = train_dl_model(model, X_tr_t, y_tr_t, epochs=150, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return prever_recursivo(model, train_scaled, test_df, scaler, target_idx, device)


# ── Visualizações ─────────────────────────────────────────────────────────────

def grafico_degradacao(todas_metricas, save_path):
    """Curva MAE × horizonte para cada modelo."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = {'XGBoost': 'tab:orange', 'Bi-LSTM': 'tab:green', 'Bi-GRU': 'tab:red'}
    marcadores = {'XGBoost': 'o', 'Bi-LSTM': 's', 'Bi-GRU': '^'}

    for modelo, metricas in todas_metricas.items():
        maes = [metricas[h][0] for h in HORIZONTES]
        ax.plot(HORIZONTES, maes, marker=marcadores[modelo], label=modelo,
                color=cores[modelo], linewidth=2, markersize=8)
        for h, mae in zip(HORIZONTES, maes):
            ax.annotate(f'{mae:.0f}', (h, mae),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=10, color=cores[modelo])

    ax.set_xlabel('Horizonte de previsão (dias)')
    ax.set_ylabel('MAE (interrupções/dia)')
    ax.set_title('Degradação do Erro (MAE) por Horizonte de Previsão')
    ax.set_xticks(HORIZONTES)
    ax.legend()
    plt.tight_layout()
    out = f"{save_path}/previsao_multihorizonte_degradacao.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Gráfico de degradação salvo: {out}")


def grafico_barras_x3(preds, y_real, datas, save_path):
    """Gráfico de barras comparando real vs modelos para x=3."""
    h = 3
    fig, ax = plt.subplots(figsize=(10, 5))
    dias = [f"t+{i+1}\n({d.strftime('%d/%m')})" for i, d in enumerate(datas[:h])]
    x = np.arange(h)
    larg = 0.18

    cores   = {'XGBoost': 'tab:orange', 'Bi-LSTM': 'tab:green', 'Bi-GRU': 'tab:red'}
    offsets = {'XGBoost': -larg, 'Bi-LSTM': 0, 'Bi-GRU': larg}

    ax.bar(x, y_real[:h], width=larg * 4.5, label='Real',
           color='tab:blue', alpha=0.25, zorder=2)

    for nome, y_pred in preds.items():
        ax.bar(x + offsets[nome], y_pred[:h], width=larg, label=nome,
               color=cores[nome], alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(dias)
    ax.set_ylabel('Interrupções diárias')
    ax.set_title('Previsão dos Próximos 3 Dias — Real vs Modelos')
    ax.legend()
    plt.tight_layout()
    out = f"{save_path}/previsao_multihorizonte_barras_x3.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Gráfico x=3 salvo: {out}")


def salvar_metricas(todas_metricas, save_path):
    rows = []
    for modelo, metricas in todas_metricas.items():
        for h, (mae_v, rmse_v) in metricas.items():
            rows.append({'Modelo': modelo, 'Horizonte': h,
                         'MAE': round(mae_v, 2), 'RMSE': round(rmse_v, 2)})
    df = pd.DataFrame(rows)
    out = f"{save_path}/previsao_multihorizonte_metricas.csv"
    df.to_csv(out, index=False)
    print(f"\nMétricas salvas: {out}")
    print(df.pivot(index='Horizonte', columns='Modelo', values='MAE')
            .rename_axis('MAE por Horizonte').to_string())


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"=== Previsão Multi-Horizonte: {HORIZONTES} dias ===\n")

    train_df, test_df = carregar(DATA_PATH)

    xgb_pred, y_real, datas, xgb_met = rodar_xgboost(train_df, test_df)
    lst_pred, lst_met                 = rodar_lstm(train_df, test_df)
    gru_pred, gru_met                 = rodar_gru(train_df, test_df)

    todas_metricas = {
        'XGBoost': xgb_met,
        'Bi-LSTM': lst_met,
        'Bi-GRU':  gru_met,
    }
    preds = {
        'XGBoost': xgb_pred,
        'Bi-LSTM': lst_pred,
        'Bi-GRU':  gru_pred,
    }

    grafico_degradacao(todas_metricas, SAVE_PATH)
    grafico_barras_x3(preds, y_real, datas, SAVE_PATH)
    salvar_metricas(todas_metricas, SAVE_PATH)

    print("\n[OK] Concluído.")
