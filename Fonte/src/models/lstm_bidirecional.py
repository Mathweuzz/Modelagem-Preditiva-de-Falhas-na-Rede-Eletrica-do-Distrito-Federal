import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class LSTM_Bidirecional(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTM_Bidirecional, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Core Bidirectional Long Short-Term Memory Block
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Fully Connected (Dense) Layers for Projection
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) # *2 due to bidirectional concat
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize internal state vectors (h0, c0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        
        # BPTT Execution over Time Steps
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Extract features from the final temporal slice (T)
        out = out[:, -1, :] 
        
        # Feedforward projection towards continuous Regression prediction
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return out

def train_lstm_model(model, X_train, y_train, epochs=200, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    losses = []
    model.train()
    print("Initiating PyTorch LSTM Training Loop...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Computational Graph)
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        # 2. Backward Pass (Gradient Descent / BPTT)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs} | Training MSE Loss: {loss.item():.5f}")
            
    print("LSTM Training Completed successfully.")
    return model, losses
