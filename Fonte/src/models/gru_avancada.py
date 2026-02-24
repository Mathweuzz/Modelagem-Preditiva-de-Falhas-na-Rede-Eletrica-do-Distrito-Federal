import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class GRU_Avancada(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRU_Avancada, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Core Bidirectional Gated Recurrent Unit Block (No Cell State)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Projection and Regularization Layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize internal hidden state vectors (h0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        
        # Run GRU over the full lookback chronological sequence
        out, hn = self.gru(x, h0.detach())
        
        # Extract features from the final temporal slice (Lags converged)
        out = out[:, -1, :] 
        
        # Multi-Layer Perceptron Mapping
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return out

def train_gru_model(model, X_train, y_train, epochs=200, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    losses = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    model.train()
    
    print("Initiating PyTorch GRU Training Loop...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward Propagation
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        # 2. Backward Propagation (BPTT / Gradients Calculation)
        loss.backward()
        optimizer.step()
        
        # 3. Dynamic Learning Rate Adjustment
        scheduler.step(loss)
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs} | GRU MSE Loss: {loss.item():.5f}")
            
    print("GRU Advanced Architectured Training Completed.")
    return model, losses
