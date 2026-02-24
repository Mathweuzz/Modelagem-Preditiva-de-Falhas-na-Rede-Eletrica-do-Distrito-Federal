import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    print("[ERRO] PyTorch não está disponível. Instale com:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    raise

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T12_modelos_dl"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

CSV_IN = DADOS_DIR / "base_diaria_interrupcoes_clima_vento.csv"

OUT_PREV = DADOS_DIR / "previsoes_dl_lstm_gru.csv"
OUT_MET = DADOS_DIR / "metricas_dl_lstm_gru.csv"
OUT_PNG = GRAFICOS_DIR / "previsao_dl_zoom_1ano.png"

# Split temporal (mesmo “estilo” da entrega 2/3: treino antes, teste depois)
CUT_DATE = pd.Timestamp("2023-09-25")  # teste a partir daqui

# Janela de lookback (dias usados para prever o dia seguinte)
LOOKBACK = 30

# Treinamento
BATCH = 64
EPOCHS = 25
LR = 1e-3
HIDDEN = 64

DEVICE = "cpu"


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


class SeqDataset(Dataset):
    def __init__(self, X, y, dates):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.dates = dates

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, cell="lstm"):
        super().__init__()
        self.cell = cell.lower()
        if self.cell == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.cell == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("cell deve ser 'lstm' ou 'gru'")
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


def load_and_build():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Não encontrei: {CSV_IN}")

    df = pd.read_csv(CSV_IN)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    # Features exógenas (diárias)
    # direção do vento como variável circular (sin/cos) — melhora muito vs usar graus “cru”
    # Converter colunas base para numérico
    base_cols = [
        "interrupcoes",
        "temperatura_media",
        "precipitacao_total_mm",
        "vento_velocidade_media_ms",
        "vento_rajada_max_ms",
        "vento_direcao_media_gr",
    ]
    for c in base_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop mínimo (data e alvo)
    df = df.dropna(subset=["data", "interrupcoes"]).copy()
    df = df.reset_index(drop=True)

    # Split temporal (precisamos dele antes para imputar com estatísticas do treino)
    train_df = df[df["data"] < CUT_DATE].copy()
    test_df  = df[df["data"] >= CUT_DATE].copy()
    if len(train_df) < LOOKBACK + 10 or len(test_df) < 10:
        raise RuntimeError("Split ficou pequeno demais. Ajuste CUT_DATE ou LOOKBACK.")

    # Imputação: preencher NaNs das features com mediana do TREINO (sem vazamento)
    fill_cols = [
        "temperatura_media",
        "precipitacao_total_mm",
        "vento_velocidade_media_ms",
        "vento_rajada_max_ms",
        "vento_direcao_media_gr",
    ]
    med = train_df[fill_cols].median(numeric_only=True)

    for c in fill_cols:
        df[c] = df[c].fillna(med[c])

    # Direção do vento -> sin/cos (agora sem NaN)
    wd = df["vento_direcao_media_gr"].to_numpy()
    wd_rad = np.deg2rad(wd)
    df["vento_dir_sin"] = np.sin(wd_rad)
    df["vento_dir_cos"] = np.cos(wd_rad)

    # Features finais (histórico inclui interrupções)
    feats = [
        "interrupcoes",
        "temperatura_media",
        "precipitacao_total_mm",
        "vento_velocidade_media_ms",
        "vento_rajada_max_ms",
        "vento_dir_sin",
        "vento_dir_cos",
    ]

    # Recalcular train/test após imputação/novas colunas
    train_df = df[df["data"] < CUT_DATE].copy()
    test_df  = df[df["data"] >= CUT_DATE].copy()

    df = df.reset_index(drop=True)

    # Split temporal
    train_df = df[df["data"] < CUT_DATE].copy()
    test_df  = df[df["data"] >= CUT_DATE].copy()

    if len(train_df) < LOOKBACK + 10 or len(test_df) < 10:
        raise RuntimeError("Split ficou pequeno demais. Ajuste CUT_DATE ou LOOKBACK.")

    # Normalização (fit só no treino!)
    feat_cols = feats[:]  # inclui interrupcoes como parte do histórico
    mu = train_df[feat_cols].mean()
    sd = train_df[feat_cols].std()

    # Se alguma std vier 0 ou NaN, evita divisão inválida
    sd = sd.replace(0, 1).fillna(1)

    df_norm = df.copy()
    df_norm[feat_cols] = (df_norm[feat_cols] - mu) / sd

    # Construir janelas: usa [t-LOOKBACK ... t-1] pra prever y[t]
    X, y, dates = [], [], []
    values = df_norm[feat_cols].to_numpy()
    y_raw = df["interrupcoes"].to_numpy()
    dts = df["data"].to_numpy()

    for i in range(LOOKBACK, len(df)):
        x_win = values[i-LOOKBACK:i, :]      # histórico até t-1
        y_t   = y_raw[i]                     # alvo em t
        dt_t  = dts[i]
        X.append(x_win)
        y.append(y_t)
        dates.append(pd.Timestamp(dt_t))

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    dates = np.array(dates)
    
    # Segurança: remover qualquer amostra com NaN/inf em X ou y
    mask_ok = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    removed = int((~mask_ok).sum())
    if removed > 0:
        print(f"[INFO] Removendo {removed} amostras com NaN/inf nas janelas.")
    X, y, dates = X[mask_ok], y[mask_ok], dates[mask_ok]

    # máscara de treino/teste no nível das amostras (pela data do alvo)
    is_train = dates < CUT_DATE
    is_test  = dates >= CUT_DATE

    X_train, y_train, d_train = X[is_train], y[is_train], dates[is_train]
    X_test,  y_test,  d_test  = X[is_test],  y[is_test],  dates[is_test]

    return X_train, y_train, d_train, X_test, y_test, d_test


def train_model(cell, X_train, y_train):
    ds = SeqDataset(X_train, y_train, None)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = RNNRegressor(input_size=X_train.shape[-1], hidden_size=HIDDEN, cell=cell).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if epoch in (1, 5, 10, 15, 20, 25):
            print(f"  [{cell.upper()}] epoch {epoch:02d}/{EPOCHS} - loss={np.mean(losses):.6f}")

    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    pred = model(xb).cpu().numpy().reshape(-1)
    return pred


def plot_zoom_1ano(df_prev: pd.DataFrame):
    # último 1 ano do teste
    df_test = df_prev[df_prev["conjunto"] == "teste"].copy()
    if df_test.empty:
        return

    end = df_test["data"].max()
    start = end - pd.Timedelta(days=365)
    z = df_test[(df_test["data"] >= start) & (df_test["data"] <= end)].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(z["data"], z["y_true"], label="Real (y)")
    plt.plot(z["data"], z["y_pred_lstm"], label="LSTM")
    plt.plot(z["data"], z["y_pred_gru"], label="GRU")
    plt.title("Previsão diária — LSTM vs GRU (zoom ~1 ano do teste)")
    plt.xlabel("Data")
    plt.ylabel("Interrupções")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()


def main():
    print("CSV_IN =", CSV_IN)
    X_train, y_train, d_train, X_test, y_test, d_test = load_and_build()
    print(f"[1] Amostras: treino={len(y_train)} | teste={len(y_test)} | LOOKBACK={LOOKBACK} | corte={CUT_DATE.date()}")

    print("[2] Treinando LSTM...")
    lstm = train_model("lstm", X_train, y_train)

    print("[3] Treinando GRU...")
    gru = train_model("gru", X_train, y_train)

    print("[4] Predizendo...")
    yhat_lstm_train = predict(lstm, X_train)
    yhat_gru_train  = predict(gru, X_train)
    yhat_lstm_test  = predict(lstm, X_test)
    yhat_gru_test   = predict(gru, X_test)

    # Métricas
    metrics = []
    for name, ytr, yhat_tr, yte, yhat_te in [
        ("LSTM", y_train, yhat_lstm_train, y_test, yhat_lstm_test),
        ("GRU",  y_train, yhat_gru_train,  y_test, yhat_gru_test),
    ]:
        metrics.append({"modelo": name, "conjunto": "treino", "MAE": mae(ytr, yhat_tr), "RMSE": rmse(ytr, yhat_tr), "R2": r2_score(ytr, yhat_tr)})
        metrics.append({"modelo": name, "conjunto": "teste",  "MAE": mae(yte, yhat_te), "RMSE": rmse(yte, yhat_te), "R2": r2_score(yte, yhat_te)})

    met = pd.DataFrame(metrics)
    met.to_csv(OUT_MET, index=False)
    print("[OK] Métricas salvas:", OUT_MET)
    print(met.to_string(index=False))

    # Previsões (para comparação e gráficos)
    df_train = pd.DataFrame({
        "data": pd.to_datetime(d_train),
        "conjunto": "treino",
        "y_true": y_train,
        "y_pred_lstm": yhat_lstm_train,
        "y_pred_gru": yhat_gru_train,
    })
    df_test = pd.DataFrame({
        "data": pd.to_datetime(d_test),
        "conjunto": "teste",
        "y_true": y_test,
        "y_pred_lstm": yhat_lstm_test,
        "y_pred_gru": yhat_gru_test,
    })
    prev = pd.concat([df_train, df_test], ignore_index=True).sort_values("data")
    prev.to_csv(OUT_PREV, index=False)
    print("[OK] Previsões salvas:", OUT_PREV)

    # Gráfico zoom 1 ano do teste
    plot_zoom_1ano(prev)
    print("[OK] Gráfico zoom 1 ano:", OUT_PNG)

    print("\n[OK] T12 concluído: LSTM + GRU com split temporal sem vazamento.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
