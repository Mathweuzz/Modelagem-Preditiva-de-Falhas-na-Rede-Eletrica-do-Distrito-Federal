"""
Previsao Multi-Horizonte — Avaliacao Direta (Direct Multi-Step)
===============================================================

Para cada horizonte h em {1, 3, 7, 14}:

  Tarefa: dados os atributos do dia t (incluindo interrupcoes[t] como
  informacao historica), prever interrupcoes[t+h].

  Protocolo:
  - Todos os modelos recebem a mesma informacao historica ate o dia t.
  - A divisao treino/teste e feita pela data-alvo (nao pela data de origem).
  - Conjunto de teste = dias-alvo em [01/06/2024, 31/05/2025] = 365 datas.
  - n = 365 para todos os modelos e horizontes (assercao verificada).

  XGBoost:
    - X de cada linha inclui interrupcoes[t] como preditor historico.
    - Alvo = interrupcoes.shift(-h) — deslocado h passos a frente.
    - Hiperparametros carregados de xgboost_best_params.json.

  Bi-LSTM / Bi-GRU:
    - Janela de entrada: [t-13,...,t] (14 dias, includindo interrupcoes[t]).
    - Alvo: interrupcoes[t+h].
    - Sequencias criadas sobre o dataset completo; divisao por data-alvo.
    - Scaler ajustado apenas sobre dias anteriores a TEST_START.

Saidas:
  results/ml/predictions_all.csv
  results/ml/metrics_multihorizon.csv
  results/ml/previsao_multihorizonte_metricas.csv

Criterio de aceite:
  Para todos os modelos e horizontes:
    - data_alvo minima = 2024-06-01
    - data_alvo maxima = 2025-05-31
    - n = 365
    - data_alvo == data_origem + h dias
"""

import os
import gc
import json
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

HORIZONTES = [1, 3, 7, 14]
SEQ_LENGTH  = 14
TARGET_COL  = 'interrupcoes'
DATA_PATH   = '../../data/dataset_engenharia_features.csv'
SAVE_PATH   = '../../results/ml'

TEST_START = pd.Timestamp('2024-06-01')
TEST_END   = pd.Timestamp('2025-05-31')


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

    params_path = f"{SAVE_PATH}/xgboost_best_params.json"
    with open(params_path) as f:
        best_params = json.load(f)
    print(f"  Parametros carregados de {params_path}: {best_params}")

    resultados = []

    for h in HORIZONTES:
        print(f"  Horizonte h={h}...")

        # interrupcoes[t] e mantido como preditor historico (alvo = t+h)
        X = df.copy()
        y_shifted = df[TARGET_COL].shift(-h)

        valid = y_shifted.notna()
        X_v  = X[valid]
        y_v  = y_shifted[valid]

        # Divisao por DATA-ALVO
        datas_alvo  = X_v.index + pd.Timedelta(days=h)
        mask_treino = datas_alvo < TEST_START
        mask_teste  = (datas_alvo >= TEST_START) & (datas_alvo <= TEST_END)

        X_tr, y_tr = X_v[mask_treino], y_v[mask_treino]
        X_te, y_te = X_v[mask_teste],  y_v[mask_teste]
        datas_alvo_te   = datas_alvo[mask_teste]
        datas_origem_te = X_te.index

        assert len(X_te) == 365, f"h={h}: esperado 365, obtido {len(X_te)}"
        assert datas_alvo_te.min() == TEST_START, f"h={h}: data minima {datas_alvo_te.min()}"
        assert datas_alvo_te.max() == TEST_END,   f"h={h}: data maxima {datas_alvo_te.max()}"

        model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
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
# DL — previsao direta por horizonte
# Janela [t-13,...,t] -> alvo t+h
# Sequencias criadas sobre o dataset completo; split por data-alvo.
# ---------------------------------------------------------------------------

def criar_sequencias_diretas(df, scaled, target_idx, seq_length, h):
    """
    Retorna (X, y, origins, targets) com datas reais de df.index.

    Sequencia i:
      X[i] = scaled[i : i+seq_length]            (janela terminando em df.index[i+seq_length-1])
      y[i] = scaled[i+seq_length+h-1, target_idx] (alvo em df.index[i+seq_length+h-1])
    """
    xs, ys, origins, targets = [], [], [], []
    n = len(df)
    for i in range(n - seq_length - h + 1):
        origin_pos = i + seq_length - 1   # ultimo dia da janela de entrada
        target_pos = origin_pos + h        # dia-alvo

        xs.append(scaled[i : origin_pos + 1])
        ys.append(scaled[target_pos, target_idx])
        origins.append(df.index[origin_pos])
        targets.append(df.index[target_pos])

    return (
        np.array(xs),
        np.array(ys),
        pd.DatetimeIndex(origins),
        pd.DatetimeIndex(targets),
    )


def rodar_dl_direto(df, ModelClass, TrainFn, nome):
    print(f"\n=== {nome} (direto) ===")
    resultados = []

    for h in HORIZONTES:
        print(f"  Horizonte h={h}...")
        set_seeds(SEED)

        # Scaler ajustado exclusivamente sobre dados de treino
        train_mask = df.index < TEST_START
        scaler = MinMaxScaler()
        scaler.fit(df[train_mask])
        scaled_all = scaler.transform(df)

        target_idx = df.columns.get_loc(TARGET_COL)

        # Sequencias sobre dataset completo
        X_all, y_all, origins_all, targets_all = criar_sequencias_diretas(
            df, scaled_all, target_idx, SEQ_LENGTH, h
        )

        # Split por data-alvo
        mask_train = targets_all < TEST_START
        mask_test  = (targets_all >= TEST_START) & (targets_all <= TEST_END)

        X_tr_np = X_all[mask_train]
        y_tr_np = y_all[mask_train]
        X_te_np = X_all[mask_test]
        y_te_np = y_all[mask_test]
        datas_origem_te = origins_all[mask_test]
        datas_alvo_te   = targets_all[mask_test]

        assert len(X_te_np) == 365, f"h={h}: esperado 365, obtido {len(X_te_np)}"
        assert datas_alvo_te.min() == TEST_START, f"h={h}: data minima {datas_alvo_te.min()}"
        assert datas_alvo_te.max() == TEST_END,   f"h={h}: data maxima {datas_alvo_te.max()}"
        delta = (datas_alvo_te - datas_origem_te).days
        assert (delta == h).all(), f"h={h}: delta de datas inconsistente"

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

        n_cols = df.shape[1]
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

        del model, X_tr_t, y_tr_t, X_tr_np, y_tr_np, X_te_np, y_te_np
        del scaled_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return resultados


# ---------------------------------------------------------------------------
# Validacao pos-execucao
# ---------------------------------------------------------------------------

def validar_predictions(df_pred):
    print("\n=== Validacao das predicoes ===")
    ok = True
    for (modelo, h), grp in df_pred.groupby(['modelo', 'horizonte']):
        n = len(grp)
        dmin = pd.to_datetime(grp['data_alvo']).min()
        dmax = pd.to_datetime(grp['data_alvo']).max()
        delta_ok = (
            pd.to_datetime(grp['data_alvo']) - pd.to_datetime(grp['data_origem'])
        ).dt.days.eq(h).all()
        status = "OK" if (n == 365 and dmin == TEST_START and dmax == TEST_END and delta_ok) else "ERRO"
        if status == "ERRO":
            ok = False
        print(f"  [{status}] {modelo} h={h}: n={n}, alvo [{dmin.date()} -> {dmax.date()}], delta_ok={delta_ok}")

    # Mesmo vetor real para todos os modelos no mesmo horizonte
    for h, grp in df_pred.groupby('horizonte'):
        tab = grp.pivot(index='data_alvo', columns='modelo', values='y_real')
        max_diff = (tab.max(axis=1) - tab.min(axis=1)).abs().max()
        status = "OK" if max_diff < 1e-3 else "AVISO"
        print(f"  [{status}] h={h}: max diferenca no y_real entre modelos = {max_diff:.6f}")

    if not ok:
        raise ValueError("Falha na validacao das previsoes multi-horizonte — verifique os resultados acima.")
    print("[OK] Todas as assercoes passaram.")


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

    # Metricas agregadas
    rows = []
    for (modelo, h), grp in df_pred.groupby(['modelo', 'horizonte']):
        m = calcular_metricas(grp['y_real'].values, grp['y_pred'].values)
        rows.append({'Modelo': modelo, 'Horizonte': h, **m})

    df_met = pd.DataFrame(rows).sort_values(['Modelo', 'Horizonte'])
    df_met.to_csv(f"{SAVE_PATH}/metrics_multihorizon.csv", index=False)
    df_met[['Modelo', 'Horizonte', 'MAE', 'RMSE']].to_csv(
        f"{SAVE_PATH}/previsao_multihorizonte_metricas.csv", index=False
    )

    print(f"Metricas -> {SAVE_PATH}/metrics_multihorizon.csv")
    print("\n=== RESUMO ===")
    print(df_met.to_string(index=False))

    validar_predictions(df_pred)
    print("\n[OK] Analise multi-horizonte concluida.")
