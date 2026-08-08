"""
GRU Bidirecional — Arquitetura, Treinamento e Avaliação
========================================================

Treina uma GRU bidirecional para previsão de interrupções diárias e
gera os artefatos:

  - results/ml/learning_curve_gru_bidirecional.png  (perda MSE por época)
  - results/ml/ts_pred_gru_bi.png                   (real vs previsto)
  - results/ml/scatter_pred_gru_bi.png              (dispersão)
  - results/ml/metrics_gru_bi.csv                   (MAE, RMSE, R², MAPE)

Fonte de dados:
  ../../data/dataset_engenharia_features.csv

Dependências internas:
  - data_loader_dl.prepare_data_dl
  - baseline_xgboost.evaluate_and_plot
  - lstm_bidirecional.reverse_scaling, plot_loss

Execução:
  cd Fonte/src/models && python gru_avancada.py
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader_dl import prepare_data_dl
from baseline_xgboost import evaluate_and_plot
from lstm_bidirecional import reverse_scaling, plot_loss, set_seeds, SEED


class AdvancedGRU(nn.Module):
    """GRU bidirecional com cabeça MLP para regressão.

    GRU é mais leve que a LSTM por não manter um cell state separado;
    em séries pequenas costuma convergir mais rápido.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout_rate=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate, bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        _, h_n = self.gru(x, h0)
        rep = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.fc1(rep)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train_gru_model(model, X_train, y_train, epochs=150, batch_size=32, lr=0.001):
    print("Iniciando Treinamento da GRU Bidirecional...")
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
        batch_size=batch_size, shuffle=False
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


if __name__ == "__main__":
    set_seeds(SEED)
    os.makedirs('../../results/ml', exist_ok=True)
    data_path = '../../data/dataset_engenharia_features.csv'

    SEQ_LENGTH = 14
    (X_train_t, y_train_t, X_test_t, y_test_t,
     scaler, target_idx, test_dates) = prepare_data_dl(data_path,
                                                       seq_length=SEQ_LENGTH)

    input_dim = X_train_t.shape[2]
    model = AdvancedGRU(input_size=input_dim, hidden_size=64, num_layers=2,
                        output_size=1, dropout_rate=0.4)

    model, losses = train_gru_model(model, X_train_t, y_train_t,
                                    epochs=150, batch_size=32)
    plot_loss(losses, "GRU_Bidirecional", "../../results/ml")

    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_preds = model(X_test_t.to(device)).cpu().numpy()

    y_pred_real = reverse_scaling(test_preds, scaler, target_idx, input_dim)
    y_test_real = reverse_scaling(y_test_t.numpy(), scaler, target_idx, input_dim)

    y_test_series = pd.Series(y_test_real, index=test_dates)

    class _PredictionWrapper:
        def predict(self, _X): return y_pred_real

    evaluate_and_plot(_PredictionWrapper(), X_test=None, y_test=y_test_series,
                      model_name='GRU_Bi', save_path='../../results/ml')
