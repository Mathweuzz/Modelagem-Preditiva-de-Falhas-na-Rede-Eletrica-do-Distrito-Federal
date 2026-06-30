"""
LSTM Bidirecional — Arquitetura, Treinamento e Avaliação
=========================================================

Treina uma LSTM bidirecional para previsão de interrupções diárias e
gera os artefatos:

  - results/ml/learning_curve_lstm_bidirecional.png  (perda MSE por época)
  - results/ml/ts_pred_lstm_bi.png                   (real vs previsto)
  - results/ml/scatter_pred_lstm_bi.png              (dispersão)
  - results/ml/metrics_lstm_bi.csv                   (MAE, RMSE, R², MAPE)

Fonte de dados:
  ../../data/dataset_engenharia_features.csv

Dependências internas:
  - data_loader_dl.prepare_data_dl
  - baseline_xgboost.evaluate_and_plot

Execução:
  cd Fonte/src/models && python lstm_bidirecional.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader_dl import prepare_data_dl
from baseline_xgboost import evaluate_and_plot


class AdvancedLSTM(nn.Module):
    """LSTM bidirecional com cabeça MLP para regressão."""

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_rate=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate, bidirectional=True
        )

        # *2 porque a saída concatena as duas direções
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]   # último passo da sequência

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train_dl_model(model, X_train, y_train, epochs=150, batch_size=32, lr=0.001):
    print("Iniciando Treinamento da LSTM Bidirecional...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                    patience=15, factor=0.5)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=False  # série temporal => sem shuffle
    )

    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # gradient clipping previne explosão do gradiente em LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Época [{epoch+1}/{epochs}] | Loss MSE: {avg_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model, loss_history


def reverse_scaling(predictions, scaler, target_idx, n_cols):
    """Inverte o MinMaxScaler para voltar à escala original (interrupções/dia)."""
    dummy = np.zeros((len(predictions), n_cols))
    dummy[:, target_idx] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


def plot_loss(loss_history, model_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss (MSE)', color='darkred')
    plt.title(f'Curva de Aprendizado - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE Norm.)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/learning_curve_{model_name.lower()}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs('../../results/ml', exist_ok=True)
    data_path = '../../data/dataset_engenharia_features.csv'

    SEQ_LENGTH = 14   # janela de 14 dias de histórico
    (X_train_t, y_train_t, X_test_t, y_test_t,
     scaler, target_idx, test_dates) = prepare_data_dl(data_path,
                                                       seq_length=SEQ_LENGTH)

    input_dim = X_train_t.shape[2]
    model = AdvancedLSTM(input_size=input_dim, hidden_size=64, num_layers=2,
                         output_size=1, dropout_rate=0.4)

    model, losses = train_dl_model(model, X_train_t, y_train_t,
                                   epochs=150, batch_size=32)
    plot_loss(losses, "LSTM_Bidirecional", "../../results/ml")

    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_preds = model(X_test_t.to(device)).cpu().numpy()

    y_pred_real = reverse_scaling(test_preds, scaler, target_idx, input_dim)
    y_test_real = reverse_scaling(y_test_t.numpy(), scaler, target_idx, input_dim)

    y_test_series = pd.Series(y_test_real, index=test_dates)

    class _PredictionWrapper:
        """Permite reaproveitar `evaluate_and_plot` (que espera .predict)."""
        def predict(self, _X): return y_pred_real

    evaluate_and_plot(_PredictionWrapper(), X_test=None, y_test=y_test_series,
                      model_name='LSTM_Bi', save_path='../../results/ml')
