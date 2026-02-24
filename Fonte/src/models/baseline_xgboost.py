import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def prepare_xgb_data(filepath, target_col='interrupcoes', test_size_days=365):
    """
    Loads and splits the continuous DataFrame for XGBoost regression.
    XGBoost does not require 3D Tensors or MinMax scaling inherently,
    as Decision Trees are invariant to monotonic transformations.
    """
    df = pd.read_csv(filepath, index_col='data', parse_dates=True)
    
    # Chronological Split (Strictly avoiding look-ahead bias)
    split_index = len(df) - test_size_days
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # Isolate Features (X) and Target (y)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test

def train_xgboost_gridsearch(X_train, y_train):
    """
    Executes a rigorous Hyperparameter Tuning via TimeSeriesSplit.
    This guarantees the Validation set is always temporally ahead of the Training set.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror', 
        tree_method='hist', # Fast histogram optimized for numerical density
        random_state=42
    )
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.5, 1],      # L1 Regularization (Lasso)
        'reg_lambda': [1, 2, 5]        # L2 Regularization (Ridge)
    }
    
    print("Initiating XGBoost GridSearch CV over Time-Series Folds...")
    grid_search = GridSearchCV(
        estimator=xgb_reg, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Optimal Hyperparameters Discovered: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_baseline(model, X_test, y_test):
    """
    Computes absolute evaluation metrics strictly on the unseen Out-of-Sample Test set.
    """
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("-" * 40)
    print("XGBoost Test Set Performance Metrics:")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R-Squared (Explained Variance): {r2:.4f}")
    print("-" * 40)
    
    return predictions, mae, rmse, r2
