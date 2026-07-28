"""
Previsao Multi-Horizonte — Avaliacao Direta (Direct Multi-Step)
===============================================================

Para cada horizonte h em {1, 3, 7, 14}:

  - Alvo: interrupcoes deslocadas h passos adiante (shift(-h))
  - Features: estado do dia t (causais, disponiveis no fim do dia t)
  - Treino: dias cuja DATA-ALVO e anterior ao periodo de teste
  - Teste: 365 dias-alvo (01/06/2024 a 31/05/2025), mesmos para todos os modelos

Estrategia direta (sem recursao) para TODOS os modelos:
  - XGBoost: ajuste direto com target deslocado
  - Bi-LSTM / Bi-GRU: janela termina em t, target e t+h (nao t+1)
    Implementado via deslocamento do vetor-alvo na construcao das sequencias.

Saidas:
  results/ml/predictions_all.csv         (modelo, data_origem, data_alvo, horizonte, y_real, y_pred)
  results/ml/metrics_multihorizon.csv    (modelo, horizonte, n, MAE, RMSE, R2, MAPE)
  results/ml/previsao_multihorizonte_metricas.csv  (alias para compatibilidade com plot_multihorizonte.py)

Execucao:
  cd Fonte/src/models && python previsao_multihorizonte.py
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from lstm_bidirecional import AdvancedLSTM, train_dl_model, reverse_scaling, set_seeds, SEED
from gru_avancada import AdvancedGRU, train_gru_model

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 12})

HORIZONTES    = [1, 3, 7, 14]
SEQ_LENGTH    = 14
TARGET_COL    = 'interrupcoes'
TEST_SIZE     = 365
DATA_PATH     = '../../data/dataset_engenharia_features.csv'
SAVE_PATH     = '../../results/ml'

TEST_START    = pd.Timestamp('2024-06-01')


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def calcular_metricas(y_true, y_pred):
    return {
        'n':    len(y_true),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2':   r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }


def carregar_dataset():
    df = pd.read_csv(DATA_PATH, index_col='data', parse_dates=True)
    print(f"Dataset: {len(df)} dias ({df.index.min().date()} -> {df.index.max().date()})")
    return df


# ---------------------------------------------------------------------------
# XGBoost — previsao direta por horizonte
# ---------------------------------------------------------------------------

def rodar_xgboost_direto(df):
    print("\n=== XGBoost (direto) ===")
    resultados = []

    for h in HORIZONTES:
        print(f"  Horizonte h={h}...")
        X = df.drop(columns=[TARGET_COL]).copy()
        y_shifted = df[TARGET_COL].shift(-h)  # alvo = interrupcoes[t+h]

        # Remover linhas onde o alvo e NaN (ultimos h dias)
        valid = y_shifted.notna()
        X_v   = X[valid]
        y_v   = y_shifted[valid]

        # Separar por data-alvo: treino = data_alvo < TEST_START
        # data_alvo do dia t = t + h dias
        datas_alvo = X_v.index + pd.Timedelta(days=h)
        mask_treino = datas_alvo < TEST_START
        mask_teste  = datas_alvo >= TEST_START

        X_tr, y_tr = X_v[mask_treino], y_v[mask_treino]
        X_te, y_te = X_v[mask_teste],  y_v[mask_teste]
        datas_alvo_te = datas_alvo[mask_teste]
        datas_origem_te = X_te.index

        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective='reg:squarederror', n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = np.maximum(0, model.predict(X_te))

        for origem, alvo, real, pred in zip(
            datas_origem_te, datas_alvo_te, y_te.values, y_pred
        ):
            resultados.append({
                'modelo': 'XGBoost', 'horizonte': h,
                'data_origem': origem.date(), 'data_alvo': alvo.date(),
                'y_real': real, 'y_pred': pred,
            })

        m = calcular_metricas(y_te.values, y_pred)
        print(f"    n={m['n']} | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f} | "
              f"R2={m['R2']:.3f} | MAPE={m['MAPE']:.2f}%")

    return resultados


# ---------------------------------------------------------------------------
# DL — previsao direta por horizonte (janela [t-13..t] -> y[t+h])
# ---------------------------------------------------------------------------

def criar_sequencias_diretas(data_scaled, target_idx, seq_length, h):
    """
    X[i] = data[i : i+seq_length]   (janela terminando em t)
    y[i] = data[i + seq_length + h - 1, target_idx]  (alvo em t+h)
    """
    xs, ys = [], []
    for i in range(len(data_scaled) - seq_length - h + 1):
        xs.append(data_scaled[i : i + seq_length])
        ys.append(data_scaled[i + seq_length + h - 1, target_idx])
    return np.array(xs), np.array(ys)


def rodar_dl_direto(df, ModelClass, TrainFn, nome):
    print(f"\n=== {nome} (direto) ===")
    resultados = []

    for h in HORIZONTES:
        print(f"  Horizonte h={h}...")
        set_seeds(SEED)

        # Scaler ajustado SEM os ultimos TEST_SIZE dias
        split_idx = len(df) - TEST_SIZE
        train_df  = df.iloc[:split_idx]
        test_df   = df.iloc[split_idx:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled  = scaler.transform(test_df)

        # Treino: sequencias diretas dentro do treino
        X_tr_np, y_tr_np = criar_sequencias_diretas(
            train_scaled, train_df.columns.get_loc(TARGET_COL), SEQ_LENGTH, h
        )

        # Teste: usar contexto (ultimos seq_length dias do treino) + test
        contexto    = train_scaled[-SEQ_LENGTH:]
        bloco_teste = np.vstack([contexto, test_scaled])
        X_te_np, y_te_np = criar_sequencias_diretas(
            bloco_teste, train_df.columns.get_loc(TARGET_COL), SEQ_LENGTH, h
        )

        # As datas de origem: primeira janela de teste começa em test_df.index[0]
        # X_te_np[i] usa bloco_teste[i:i+SEQ_LENGTH]
        # bloco_teste[0:SEQ_LENGTH] = contexto => origem = test_df.index[0]
        # bloco_teste[1:SEQ_LENGTH+1] = contexto[1:]+test[0] => origem = test_df.index[0]+1 dia
        n_te = len(X_te_np)
        datas_origem_te = test_df.index[:n_te]
        datas_alvo_te   = datas_origem_te + pd.Timedelta(days=h)

        # Converter para tensores
        X_tr_t = torch.from_numpy(X_tr_np).float()
        y_tr_t = torch.from_numpy(y_tr_np).float().unsqueeze(1)

        input_dim = X_tr_t.shape[2]
        model = ModelClass(input_size=input_dim, hidden_size=64, num_layers=2,
                           output_size=1, dropout_rate=0.4)
        model, _ = TrainFn(model, X_tr_t, y_tr_t, epochs=150, batch_size=32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            preds_norm = model(
                torch.from_numpy(X_te_np).float().to(device)
            ).cpu().numpy().flatten()

        # Inverter escala
        n_cols = train_df.shape[1]
        target_idx = train_df.columns.get_loc(TARGET_COL)
        dummy = np.zeros((len(preds_norm), n_cols))
        dummy[:, target_idx] = preds_norm
        y_pred_real = np.maximum(0, scaler.inverse_transform(dummy)[:, target_idx])

        dummy2 = np.zeros((len(y_te_np), n_cols))
        dummy2[:, target_idx] = y_te_np
        y_real_real = scaler.inverse_transform(dummy2)[:, target_idx]

        for origem, alvo, real, pred in zip(
            datas_origem_te, datas_alvo_te, y_real_real, y_pred_real
        ):
            resultados.append({
                'modelo': nome, 'horizonte': h,
                'data_origem': origem.date(), 'data_alvo': alvo.date(),
                'y_real': real, 'y_pred': pred,
            })

        m = calcular_metricas(y_real_real, y_pred_real)
        print(f"    n={m['n']} | MAE={m['MAE']:.2f} | RMSE={m['RMSE']:.2f} | "
              f"R2={m['R2']:.3f} | MAPE={m['MAPE']:.2f}%")

        # Liberar memoria entre horizontes
        del model, X_tr_t, y_tr_t, X_tr_np, y_tr_np, X_te_np, y_te_np
        del train_scaled, test_scaled, contexto, bloco_teste
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return resultados


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seeds(SEED)
    os.makedirs(SAVE_PATH, exist_ok=True)

    df = carregar_dataset()

    todos = []
    todos += rodar_xgboost_direto(df)
    todos += rodar_dl_direto(df, AdvancedLSTM, train_dl_model, 'Bi-LSTM')
    todos += rodar_dl_direto(df, AdvancedGRU,  train_gru_model, 'Bi-GRU')

    df_pred = pd.DataFrame(todos)
    df_pred.to_csv(f"{SAVE_PATH}/predictions_all.csv", index=False)
    print(f"\nPrevisoes -> {SAVE_PATH}/predictions_all.csv")

    # Metricas agregadas por modelo e horizonte
    rows = []
    for (modelo, h), grp in df_pred.groupby(['modelo', 'horizonte']):
        m = calcular_metricas(grp['y_real'].values, grp['y_pred'].values)
        rows.append({'Modelo': modelo, 'Horizonte': h, **m})

    df_met = pd.DataFrame(rows).sort_values(['Modelo', 'Horizonte'])
    df_met.to_csv(f"{SAVE_PATH}/metrics_multihorizon.csv", index=False)

    # Alias para compatibilidade com plot_multihorizonte.py
    df_met[['Modelo', 'Horizonte', 'MAE', 'RMSE']].to_csv(
        f"{SAVE_PATH}/previsao_multihorizonte_metricas.csv", index=False
    )

    print(f"Metricas -> {SAVE_PATH}/metrics_multihorizon.csv")
    print("\n=== RESUMO ===")
    print(df_met.to_string(index=False))
    print("\n[OK] Analise multi-horizonte concluida.")
