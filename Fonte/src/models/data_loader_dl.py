import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os

def create_sequences(data, target_idx, seq_length):
    """
    Creates temporal sliding windows for Deep Learning architectures.
    X tensor: (Batch, Sequence Length, Features)
    Y tensor: (Batch, 1)
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data_dl(filepath, target_col='interrupcoes', seq_length=14, test_size_days=365):
    """
    Loads, scales, and splits the time series strictly avoiding Data Leakage.
    The MinMaxScaler is fitted ONLY on the training partition.
    """
    print("Loading CSV into Pandas DataFrame...")
    df = pd.read_csv(filepath, index_col='data', parse_dates=True)
    target_idx = df.columns.get_loc(target_col)
    
    # Chronological Split (No random K-Fold to preserve temporal inertia)
    split_index = len(df) - test_size_days
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # Scale variables into [0, 1] range for Gradient Descent stability
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    print("Applying Sliding Window logic...")
    X_train, y_train = create_sequences(train_scaled, target_idx, seq_length)
    X_test, y_test = create_sequences(test_scaled, target_idx, seq_length)
    
    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
    
    test_dates = test_df.index[seq_length:]
    
    print(f"Train Tensor: {X_train_tensor.shape}")
    print(f"Test Tensor: {X_test_tensor.shape}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler, target_idx, test_dates
