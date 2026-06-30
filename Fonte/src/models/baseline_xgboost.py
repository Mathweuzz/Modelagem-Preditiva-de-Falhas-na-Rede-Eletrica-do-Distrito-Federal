"""
Baseline XGBoost — Treinamento, Avaliação e Geração de Gráficos
================================================================

Treina um regressor XGBoost para previsão de interrupções diárias e
gera os artefatos:

  - results/ml/ts_pred_xgboost.png            (real vs previsto, série temporal)
  - results/ml/scatter_pred_xgboost.png       (real vs previsto, dispersão)
  - results/ml/feature_importance_xgboost.png (importância das 15 melhores features)
  - results/ml/metrics_xgboost.csv            (MAE, RMSE, R², MAPE)

Fonte de dados:
  ../../data/dataset_engenharia_features.csv  (saída do 03_feature_engineering.py)

Execução:
  cd Fonte/src/models && python baseline_xgboost.py
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 12})


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # epsilon evita divisão por zero quando y_true == 0
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def load_and_split_data(filepath, target_col='interrupcoes', test_size_days=365):
    print("Carregando dataset com features avançadas...")
    df = pd.read_csv(filepath, index_col='data', parse_dates=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Time Series Split (chronological — sem data leakage)
    split_index = len(df) - test_size_days
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    print("Treinando modelo XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    return model


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
        plt.figure(figsize=(12, 10))
        xgb.plot_importance(model, max_num_features=15, height=0.5,
                            title=f'Importância das Variáveis (Top 15) - {model_name}',
                            color='tab:orange')
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance_{model_name.lower()}.png", dpi=300)
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
    os.makedirs('../../results/ml', exist_ok=True)

    data_path = '../../data/dataset_engenharia_features.csv'

    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path, test_size_days=365
    )

    xgb_model = train_xgboost(X_train, y_train)

    evaluate_and_plot(xgb_model, X_test, y_test,
                      model_name='XGBoost', save_path='../../results/ml')
