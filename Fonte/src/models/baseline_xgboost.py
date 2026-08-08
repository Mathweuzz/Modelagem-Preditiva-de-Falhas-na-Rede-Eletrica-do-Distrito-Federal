"""
Baseline XGBoost — Previsao Direta h=1 (One-Step-Ahead)
=========================================================

Treina um regressor XGBoost para previsao de interrupcoes do DIA SEGUINTE
(h=1 direto), usando todas as features do dia t como entrada — incluindo
interrupcoes[t], que é informacao historica quando o alvo é interrupcoes[t+1].

Isso torna a avaliacao principal equivalente ao horizonte h=1 do
previsao_multihorizonte.py, permitindo comparacao justa com as RNNs.

Divisao treino/teste por DATA-ALVO:
  - Treino : pares (t, t+1) onde t+1 < 2024-06-01
  - Teste  : pares (t, t+1) onde 2024-06-01 <= t+1 <= 2025-05-31  (365 dias)

Artefatos gerados:
  - results/ml/ts_pred_xgboost.png
  - results/ml/scatter_pred_xgboost.png
  - results/ml/feature_importance_xgboost.png
  - results/ml/metrics_xgboost.csv
  - results/ml/predictions_xgboost.csv
  - results/ml/xgboost_best_params.json  (consumido por previsao_multihorizonte.py)
  - results/ml/xgboost_cv_results.csv

Execucao:
  cd Fonte/src/models && python baseline_xgboost.py
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 12})


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 \
        if nonzero.any() else np.nan


TEST_START = pd.Timestamp('2024-06-01')
TEST_END   = pd.Timestamp('2025-05-31')


def load_and_split_data(filepath, target_col='interrupcoes', test_size_days=365):
    """
    Carrega o dataset e prepara divisao treino/teste para previsao h=1.

    - X[t] inclui interrupcoes[t] como preditor historico.
    - y[t] = interrupcoes[t+1] (deslocado 1 passo a frente).
    - Divisao por data-alvo (t+1), nao por posicao de linha.
    - Conjunto de teste: 365 dias-alvo em [01/06/2024, 31/05/2025].
    """
    print("Carregando dataset com features avancadas...")
    df = pd.read_csv(filepath, index_col='data', parse_dates=True)

    # Previsao verdadeira h=1: alvo = interrupcoes do dia seguinte
    X = df.copy()                           # inclui interrupcoes[t] como feature historica
    y_shifted = df[target_col].shift(-1)    # alvo = interrupcoes[t+1]

    # Remover ultima linha (sem alvo)
    valid = y_shifted.notna()
    X         = X[valid]
    y_shifted = y_shifted[valid]

    # Split por data-alvo
    datas_alvo  = X.index + pd.Timedelta(days=1)
    mask_train  = datas_alvo < TEST_START
    mask_test   = (datas_alvo >= TEST_START) & (datas_alvo <= TEST_END)

    X_train = X[mask_train]
    X_test  = X[mask_test]
    y_train = y_shifted[mask_train]
    y_test  = pd.Series(
        y_shifted[mask_test].values,
        index=datas_alvo[mask_test],
        name=target_col,
    )

    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    assert len(X_test) == 365, f"Esperado 365 amostras de teste, obtido {len(X_test)}"
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, save_path='../../results/ml'):
    """
    Treina XGBoost com Grid Search temporal (TimeSeriesSplit, 5 folds).
    Salva cv_results.csv e best_params.json em save_path.
    """
    print("Iniciando Grid Search com TimeSeriesSplit (5 folds)...")

    param_grid = {
        'n_estimators':     [300, 500],
        'learning_rate':    [0.03, 0.05, 0.1],
        'max_depth':        [4, 6, 8],
        'subsample':        [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [1, 3],
        'gamma':            [0, 0.1],
    }

    base = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    print(f"\nMelhores hiperparametros: {best_params}")
    print(f"MAE CV (melhor): {-gs.best_score_:.2f}")

    # Salvar artefatos
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame(gs.cv_results_).to_csv(
        f"{save_path}/xgboost_cv_results.csv", index=False
    )
    with open(f"{save_path}/xgboost_best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"CV results -> {save_path}/xgboost_cv_results.csv")
    print(f"Best params -> {save_path}/xgboost_best_params.json")

    return gs.best_estimator_


def evaluate_and_plot(model, X_test, y_test, model_name, save_path):
    """
    Função de avaliação genérica usada por XGBoost, LSTM-Bi e GRU-Bi.
    Os modelos PyTorch passam X_test=None e usam um wrapper FakeModel
    com `.predict` retornando as previsões já calculadas (ver lstm/gru).
    """
    print(f"Avaliando performance de {model_name}...")
    y_pred = model.predict(X_test)

    # Interrupções são contagens >= 0; clipamos negativos
    y_pred = np.maximum(0, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("-" * 30)
    print(f"Métricas Globais - {model_name}:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 30)

    # Real vs Previsto — Série Temporal
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test.values, label='Valores Reais (Test)',
             color='tab:blue', alpha=0.7)
    plt.plot(y_test.index, y_pred, label=f'Previsão ({model_name})',
             color='tab:red', alpha=0.9, linewidth=2)
    plt.title(f'Série Temporal de Teste: Real vs Previsto - {model_name}')
    plt.xlabel('Data')
    plt.ylabel('Quantidade de Interrupções Diárias')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/ts_pred_{model_name.lower()}.png", dpi=300)
    plt.close()

    # Real vs Previsto — Dispersão
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='tab:purple')
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p2, p1], [p2, p1], 'k--', lw=2,
             label=r'Predição Perfeita ($y = \hat{y}$)')
    plt.title(f'Dispersão: Real vs Previsto - {model_name}')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{save_path}/scatter_pred_{model_name.lower()}.png", dpi=300)
    plt.close()

    # Feature Importance — só faz sentido pra modelos baseados em árvores
    if hasattr(model, 'get_booster'):
        fig, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(model, ax=ax, max_num_features=15, height=0.5,
                            importance_type='gain',
                            show_values=False,
                            color='tab:orange')
        ax.set_title('Importância por Ganho Médio — XGBoost (Top 15)')
        ax.set_xlabel('Ganho médio')
        ax.set_ylabel('Variável')
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance_{model_name.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Métricas em CSV (consumidas pela tabela do Cap. 4)
    metrics_df = pd.DataFrame([{
        'Model': model_name,
        'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape
    }])
    metrics_df.to_csv(f"{save_path}/metrics_{model_name.lower()}.csv", index=False)

    # Previsões em CSV (consumidas pelo advanced_plots.py para gráficos comparativos)
    pd.DataFrame(
        {'real': y_test.values, 'pred': y_pred},
        index=y_test.index
    ).rename_axis('data').to_csv(f"{save_path}/predictions_{model_name.lower()}.csv")

    print(f"Arquivos gerados em {save_path}/")


if __name__ == "__main__":
    SAVE_PATH = '../../results/ml'
    os.makedirs(SAVE_PATH, exist_ok=True)

    data_path = '../../data/dataset_engenharia_features.csv'

    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path, test_size_days=365
    )

    xgb_model = train_xgboost(X_train, y_train, save_path=SAVE_PATH)

    evaluate_and_plot(xgb_model, X_test, y_test,
                      model_name='XGBoost', save_path=SAVE_PATH)
