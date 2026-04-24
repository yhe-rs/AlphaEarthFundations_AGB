#!/usr/bin/env python
# coding: utf-8

import platform
import psutil
import GPUtil

system_info = platform.uname()

print("\n******System Information:")
print(f"System: {system_info.system}")
print(f"Node Name: {system_info.node}")
print(f"Release: {system_info.release}")
print(f"Version: {system_info.version}")
print(f"Machine: {system_info.machine}")
print(f"Processor: {system_info.processor}")

cpu_info = platform.processor()
cpu_count = psutil.cpu_count(logical=False)
logical_cpu_count = psutil.cpu_count(logical=True)

print("\n******CPU Information:")
print(f"Processor: {cpu_info}")
print(f"Physical Cores: {cpu_count}")
print(f"Logical Cores: {logical_cpu_count}")

memory_info = psutil.virtual_memory()

print("\n******Memory Information:")
print(f"Total Memory: {memory_info.total} bytes")
print(f"Available Memory: {memory_info.available} bytes")
print(f"Used Memory: {memory_info.used} bytes")
print(f"Memory Utilization: {memory_info.percent}%")

disk_info = psutil.disk_usage('/')

print("\n******Disk Information:")
print(f"Total Disk Space: {disk_info.total} bytes")
print(f"Used Disk Space: {disk_info.used} bytes")
print(f"Free Disk Space: {disk_info.free} bytes")
print(f"Disk Space Utilization: {disk_info.percent}%")


from eBoruta import eBoruta
# import geopandas as gpd
import pandas as pd
import os
import numpy as np
import time
from glob import glob
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
import plotly.io as pio
import shap
import datetime
import math
import joblib

from plotly.io import show
from optuna.importance import get_param_importances, MeanDecreaseImpurityImportanceEvaluator
from datetime import datetime
# display(HTML("<style>.container { width:80% !important; }</style>"))
# pd.set_option("display.max_colwidth", 100)

import io
from contextlib import redirect_stdout
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    # Attempt to get GPU info
    gpus = GPUtil.getGPUs()
except (ValueError, Exception) as e:
    # This catches the "NVIDIA-SMI has failed" error specifically
    print(f"\nWarning: GPUtil could not retrieve GPU stats. (Error: {e})")
    print("This usually means NVIDIA drivers are missing, broken, or not communicating.")
    gpus = []

if not gpus:
    print("\nNo GPU detected by GPUtil.")
    # Check if PyTorch can see it even if GPUtil fails
    if torch.cuda.is_available():
        print(f"Note: PyTorch CAN see {torch.cuda.device_count()} GPU(s) despite GPUtil error.")
    else:
        print("Confirmed: PyTorch also cannot see any CUDA devices.")
else:
    for i, gpu in enumerate(gpus):
        print(f"\n******GPU {i + 1} Information:")
        print(f"ID: {gpu.id}")
        print(f"Name: {gpu.name}")
        print(f"Driver: {gpu.driver}")
        print(f"GPU Memory Total: {gpu.memoryTotal} MB")
        print(f"GPU Memory Free: {gpu.memoryFree} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        # Added rounding for cleaner output
        print(f"GPU Load: {round(gpu.load * 100, 2)}%")
        print(f"GPU Temperature: {gpu.temperature}°C")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import optuna
    
print("\n******Current working dir", os.getcwd())

print('torch=:',torch.__version__)
print('optuna=:',optuna.__version__)
# print('optunahub=:',optunahub.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())



# load and preprocess data
def prepare_df(csv_path, cols, dataset, save_dir, timestamp=12, cover_type=None):
    """
    Parameters:
    - csv_path: Path to the CSV file.
    - cols: List of column names for features.
    - dataset: train / validation / test
    - save_dir: Directory path to save preprocessed data.
    - cover_type: landcover type to filter (optional)

    Returns:
    - X_df: DataFrame of features.
    - y_series: Series of target variable.
    """

    # Load the data
    df = pd.read_csv(csv_path)

    # Optional filtering
    if cover_type is not None:
        df = df[df["Cover"] == cover_type].copy()
        print(f"Filtering for cover type: {cover_type}")
    else:
        print("Using all land cover types")

    # -----------------------------
    # Grouping columns
    # -----------------------------
    group_cols = ["Lat", "Lon", "INFyS_date", "AGBD"]

    # -----------------------------
    # Temporal ordering
    # -----------------------------
    df["quarter_order"] = df["quarter"].map({
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,        
    })

    df = df.sort_values(by=group_cols + ["quarter_order"]).reset_index(drop=True)

    X_list = []   # will collect rows that belong to valid sequences
    y_list = []

    # -----------------------------
    # Group and filter by exact timestamp length
    # -----------------------------
    for key, group in df.groupby(group_cols):
        group = group.sort_values("quarter_order")

        # Enforce fixed sequence length (consistency check)
        if len(group) != timestamp:
            continue

        # Keep all rows of this valid group
        X_list.append(group)

        # Take AGBD (same for the whole group)
        y_val = group["AGBD"].iloc[0]
        y_list.append(y_val)

    # ========================== Build output ==========================
    if len(X_list) == 0:
        print(f"Warning: No valid sequences with exactly {timestamp} timesteps found.")
        X_df = pd.DataFrame(columns=df.columns)
        y_series = pd.Series([], name="AGBD", dtype=np.float64)
    else:
        # Concatenate all valid groups back into one DataFrame
        X_df = pd.concat(X_list, ignore_index=True)

        # y as pandas Series (like df["AGBD"])
        y_series = pd.Series(y_list, name="AGBD", dtype=np.float64)

    print(f"Prepared from {csv_path}:")
    print(f"   → X_df shape : {X_df.shape}   (only groups with exactly {timestamp} timesteps)")
    print(f"   → y_series shape : {y_series.shape}   ({len(y_series)} valid samples)")
    
    X = X_df[cols]
    y = y_series
    
    return X, y



# plot hist of target via in training dataset
def plot_target_histograms(y, datatype, save_dir):
    """
    Plot a histogram of the target variable and save it as an image file.

    Parameters:
    - y: Series of target variable.
    - datatype: cha: type in train, vallidation, test  
    - save_dir: Directory path to save the plot.

    Returns:
    - None
    """
    
    # Calculate the sample counts
    train_count = len(y)
    min_value = y.min()
    max_value = y.max()

    plt.figure(figsize=(4, 3))

    # Plot histogram for training data
    plt.hist(y, bins=200, alpha=0.7, 
             label=f'Total samples (n={train_count}) \nMin value={min_value:.3f} Mg/ha \nMax value={max_value:.3f} Mg/ha', 
             color='blue')

    plt.xlabel('AGB (Mg/ha)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {datatype} set')
    plt.legend(loc='upper right', frameon=False)

    plt.savefig(f"{save_dir}hist-{datatype}.png", dpi=600, pad_inches=0.02, bbox_inches='tight')
    # plt.show(block=False)
    plt.close()

    print(f"\n⚡ Histogram of {datatype} set saved successfully!")
   
    return

def standard_df(X, y, save_dir):
    """
    Load data from a CSV file, preprocess it, and convert it into a df for rf.

    Parameters:
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X: DataFrame of features.
    - y: Series of target variable.
    """

    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Scale features (X is already 2D)
    X_scaled = feature_scaler.fit_transform(X)

    # Scale target (must be 2D)
    y_scaled = target_scaler.fit_transform(
        y.values.reshape(-1, 1)
    )

    # save scaler
    joblib.dump(feature_scaler, f'{save_dir}feature_scaler.joblib')
    joblib.dump(target_scaler, f'{save_dir}target_scaler.joblib')

    print(f"\n🚀X_scaled_shape:\n", X_scaled.shape)
    print(f"\n🚀y_scaled_shape:\n", y_scaled.shape)  
    print(f"\n🚀X.head(2):\n", X.head(2))
    print(f"\n🚀X_scaled[:2]:\n", X_scaled[:2])
    print(f"\n🚀y.head(2):\n", y.head(2))
    print(f"\n🚀y_scaled[:2]:\n", y_scaled[:2])
    print("\nDataframe StandardScaler successfully!")
    
    return X_scaled, y_scaled,  feature_scaler, target_scaler


def prepare_seq(X_scaled, y_scaled, timestamp=12):
    """
    Reshape the flat scaled data into 3D format for LSTM / Transformer.
    
    Parameters:
    - X_scaled: 2D numpy array from standard_df()  → shape (total_rows, n_features)
    - y_scaled: 2D numpy array from standard_df()  → shape (n_samples, 1)
    - timestamp: number of timesteps per sequence (default=2)
    
    Returns:
    - X_3d: numpy array of shape (n_samples, timestamp, n_features)
    - y_2d: numpy array of shape (n_samples, 1)   # ready for model.fit()
    """
    import numpy as np
    
    print(f"\n🔄 Reshaping data to 3D sequence format (timestamp={timestamp})...")
    
    # Check if inputs are numpy arrays
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = np.array(X_scaled)
    if not isinstance(y_scaled, np.ndarray):
        y_scaled = np.array(y_scaled)
    
    n_features = X_scaled.shape[1]
    n_samples = len(y_scaled)                   # Each y corresponds to one full sequence
    
    # Reshape X to 3D: (n_samples, timestamp, n_features)
    X_3d = X_scaled.reshape(n_samples, timestamp, n_features)
    
    # y_scaled is already (n_samples, 1), but we ensure it's 2D
    y_2d = y_scaled.reshape(-1, 1)
    
    print(f"✅ Successfully reshaped:")
    print(f"   X_3d shape : {X_3d.shape}   ← (n_samples, timesteps={timestamp}, features={n_features})")
    print(f"   y_2d shape  : {y_2d.shape}   ← (n_samples, 1)")
    
    # Optional: Show a small sample for verification
    print(f"\nSample of first sequence (scaled):")
    print(X_3d[0])
    
    return X_3d, y_2d


# Updated prepare_tensor (see detailed version above)
def prepare_tensor(X_scaled: np.ndarray, y_scaled: np.ndarray):
    X_tensor = torch.FloatTensor(X_scaled) # (n_samples, 2, 124)
    y_tensor = torch.FloatTensor(y_scaled)                # (n_samples, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"✅ GRU-compatible TensorDataset created → {len(dataset)} samples, "
          f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    
    return dataset

    
# Updated: Bidirectional GRU Model (BiGRU)
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # ← added for bidirectional
        
        # Critical changes for BiGRU:
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0, 
            bidirectional=True   # ← this enables bidirectional processing
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)  # ← critical fix: *2

    def forward(self, x):
        # Hidden state must account for both directions
        h0 = torch.zeros(
            self.num_layers * self.num_directions, 
            x.size(0), 
            self.hidden_size
        ).to(DEVICE)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])   # last time step (now has 2× hidden size)
        return out

    
def objective(trial, save_dir):
    # # --- 1. Define Search Space ---
    # hidden_size =  trial.suggest_categorical(f"hidden_size", [16, 32, 64, 128, 256, 512,1024])
    # num_layers = trial.suggest_int("num_layers", 1, 8, step=1)
    # initial_lr = trial.suggest_categorical("lr", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    # dropout = trial.suggest_float("dropout",  0.0, 0.6, step=0.1)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512,1024])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    # weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0])  # log scale
    # --- 1. Define Search Space ---
    hidden_size =  trial.suggest_categorical(f"hidden_size", [16, 32, 64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 6, step=1)
    initial_lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 1e-5])
    dropout = trial.suggest_float("dropout",  0.0, 0.6, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0])  # log scale
    
    # --- 2. Setup Trial Objects ---
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader =  DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    model = BiGRUModel(INPUT_SIZE, 
                     hidden_size, 
                     num_layers, 
                     OUTPUT_SIZE, 
                     dropout).to(DEVICE)
    # --- 4. Optimizer Initialization ---
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("rmsprop_momentum", 0.0, 0.99)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.0, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    criterion = nn.MSELoss()


    # === Print Model Structure ===   
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    print(model)

    print("\nParameter summary:")
    print("-"*50)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("-"*50)
    
    # Option 3: Manual detailed print
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("-"*50)

    # Scheduler: Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # reduce LR by half
        patience=20,      # wait # epochs without improvement
        # min_lr=1e-6,
    )
    
    # --- 5. Early Stopping Settings ---
    EPOCHS = 300
    patience = 30
    # min_delta = 1e-5
    best_val_loss = float('inf')
    early_stop_counter = 0

    # --- 5. Training Loop ---
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_mse_total = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_total += loss.item()

        avg_train_mse = train_mse_total / len(train_loader)
        # avg_train_rmse = math.sqrt(avg_train_mse)

        # --- Validation ---
        model.eval()
        val_mse_total = 0.0
        with torch.no_grad():
            for v_X, v_y in val_loader:
                v_X, v_y = v_X.to(DEVICE), v_y.to(DEVICE)
                v_outputs = model(v_X)
                v_loss = criterion(v_outputs, v_y)
                val_mse_total += v_loss.item()

        avg_val_mse = val_mse_total / len(val_loader)
        # avg_val_rmse = math.sqrt(avg_val_mse)

        # --- Early Stopping Logic ---
        # if avg_val_mse < best_val_loss - min_delta:
        if avg_val_mse < best_val_loss:            
            best_val_loss = avg_val_mse
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"⚠️Trial{[trial.number]} Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train | Val -> MSE LOSS: {avg_train_mse:.4f} | {avg_val_mse:.4f}")
        print("-" * 30)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_mse)

        # Early stopping check
        if early_stop_counter >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break

        # Optuna pruning
        trial.report(avg_val_mse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    print(f"\nTraining Complete. Best Validation MSE: {best_val_loss:.4f}")

    return best_val_loss    

def retrain(study, INPUT_SIZE, OUTPUT_SIZE, train_tensor, val_tensor, save_dir):
    # -----------------------------
    # Print Optuna best results
    # -----------------------------
    print("\nBest trial number:", study.best_trial.number)
    print("Best hyperparameters:", study.best_params)
    print("Best MSE:", study.best_value)
    print("Training time:", used_time)

    with open(f"{save_dir}_best_para_result.txt", "w") as file:
        file.write(f"Best trial number: {study.best_trial.number}\n")
        file.write(f"Best hyperparameters: {study.best_params}\n")
        file.write(f"Best MSE: {study.best_value}\n")
        file.write(f"Training time: {used_time}")
        
    # -----------------------------
    # 1. Retrieve Best Hyperparameters
    # -----------------------------
    best_params = study.best_params
    num_layers = best_params["num_layers"]
    batch_size = best_params["batch_size"]
    initial_lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    optimizer_name = best_params["optimizer"]
    hidden_size =  best_params["hidden_size"]
    dropout = best_params["dropout"]
    
    print("\n" + "="*30)
    print(f"🚀 RETRAINING BEST MODEL (Trial {study.best_trial.number})")
    print(f"Optimizer: {optimizer_name} | Batch Size: {batch_size}")
    print(f"Architecture: {hidden_size}")
    print("="*30)
    
    # -----------------------------
    # 2. DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_tensor,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_tensor,
        batch_size=batch_size,
        shuffle=False
    )

    # -----------------------------
    # 3. Model Initialization
    # -----------------------------
    model = BiGRUModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers = num_layers,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(DEVICE)
    
    
    # === Capture and Save Model Structure ===
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE")
        print("="*50)
        print(model)
        
        print("\nParameter summary:")
        print("-"*50)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        print("-"*50)
    
    # Also print to console (optional, for live monitoring)
    print(buffer.getvalue())
    
    # Write everything to file
    output_path = f"{save_dir}_model_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())
    print(f"Model summary saved to {output_path}")

    # -----------------------------
    # 4. Optimizer Selection (MATCHING OPTUNA LOGIC)
    # -----------------------------
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        momentum = best_params["rmsprop_momentum"]
        optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "SGD":
        momentum = best_params["sgd_momentum"]
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)

    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        # min_lr=1e-6,
    )

    # -----------------------------
    # Early stopping
    # -----------------------------
    EPOCHS = 300
    patience = 30
    # min_delta = 1e-5
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    #
    best_model_path = f"{save_dir}_final_model.pth"
    # -----------------------------
    # Loss tracking
    # -----------------------------
    epoch_list = []
    train_mse_list = []
    val_mse_list = []

    # --- 5. Training Loop ---
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_mse_total = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_total += loss.item()

        avg_train_mse = train_mse_total / len(train_loader)
        # avg_train_rmse = math.sqrt(avg_train_mse)

        # --- Validation ---
        model.eval()
        val_mse_total = 0.0
        with torch.no_grad():
            for v_X, v_y in val_loader:
                v_X, v_y = v_X.to(DEVICE), v_y.to(DEVICE)
                v_outputs = model(v_X)
                v_loss = criterion(v_outputs, v_y)
                val_mse_total += v_loss.item()

        avg_val_mse = val_mse_total / len(val_loader)
        # avg_val_rmse = math.sqrt(avg_val_mse)
        
        # Save losses
        epoch_list.append(epoch + 1)
        train_mse_list.append(avg_train_mse)
        val_mse_list.append(avg_val_mse)
        
        # --- Early Stopping Logic ---
        # if avg_val_mse < best_val_loss - min_delta:
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        print(f"⚠️Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train | Val -> MSE LOSS: {avg_train_mse:.4f} | {avg_val_mse:.4f}")
        print("-" * 30)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_mse)

        # Early stopping check
        if early_stop_counter >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break

    # -----------------------------
    # Load best model
    # -----------------------------
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # -----------------------------
    # Save CSV
    # -----------------------------
    loss_df = pd.DataFrame({
        "epoch": epoch_list,
        "train_mse": train_mse_list,
        "val_mse": val_mse_list,
    })
    loss_df["best_epoch"] = loss_df["epoch"] == best_epoch
    csv_path = f"{save_dir}/re_training_val_loss.csv"
    loss_df.to_csv(csv_path, index=False)

    # -----------------------------
    # Plot losses
    # -----------------------------
    figsize = (2 ,2)  # Adjust the figure size as needed
    scale_factor = figsize[0]
    plt.figure(figsize=figsize)
    
    plt.plot(epoch_list, 
             train_mse_list, 
             linewidth=0.7,
             label="Train loss")
    plt.plot(epoch_list, 
             val_mse_list, 
             linewidth=0.7,
             label="Validation loss")
    plt.axvline(best_epoch, 
                color="gray", 
                linestyle="--", 
                linewidth=0.7,
                alpha=0.7,
                label=f"Best Epoch {best_epoch}"
               )


    # Find the maximum value
    max_value = loss_df["epoch"].max()
    
    print("\nmax of epoch：",max_value)

    # Set x-y axis limit from 0 to top
    top = max_value
    # Round 'top' up to the nearest multiple of 100
    rounded_top = math.ceil(top / 10.0) * 10
    plt.xlim(0, rounded_top)

    # # Adjust the size of x and y axis tick labels
    plt.xticks(ticks=np.arange(0, rounded_top+1, 20),fontsize=4 * scale_factor, rotation=0)
    
    plt.xlabel("Epoch", fontsize=4* scale_factor)
    plt.ylabel("MSE", fontsize=4* scale_factor)
    # plt.title("Training & Validation Loss", fontsize=4* scale_factor)
    plt.legend(frameon=False, fontsize=3* scale_factor, loc='upper right')
    # plt.grid(True)

    plot_path = f"{save_dir}/re_training_val_loss.png"
    plt.savefig(plot_path, dpi=600, pad_inches=0.02, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # Final summary
    # -----------------------------
    print(f"\nTraining Complete.")
    print(f"Best Validation MSE: {best_val_loss:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"📄 Loss CSV saved to: {csv_path}")
    print(f"📊 Loss plot saved to: {plot_path}")

    return model


# define rRMSE func
def relative_rmse(y_true, y_pred):
    """
    Calculate the Relative Root Mean Squared Error (RRMSE).

    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values

    Returns:
    float: RRMSE value as a percentage
    """
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate the mean of the actual values
    mean_y_true = np.mean(y_true)

    # Calculate RRMSE
    rrmse = rmse / mean_y_true if mean_y_true != 0 else np.inf  # Avoid division by zero

    return rrmse * 100




# define func of scatter plot for y and y_prediction    
def evaluate_and_plot(final_model, data_tensor, feature_scaler, target_scaler, tipe, save_dir):

    final_model.eval()

    test_loader = DataLoader(
        data_tensor,
        batch_size=128,
        shuffle=True
    )

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            outputs = final_model(X_batch)
            
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    # Concatenate batches
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1, 1)
    y_true = np.concatenate(y_true_list, axis=0).reshape(-1, 1)
    
    # Inverse scaling (MUST be 2D)
    y_pred_inverse = target_scaler.inverse_transform(y_pred)
    y_true_inverse = target_scaler.inverse_transform(y_true)
    
    # Metrics
    r2 = r2_score(y_true_inverse, y_pred_inverse)
    rmse = root_mean_squared_error(y_true_inverse, y_pred_inverse)
    rrmse = relative_rmse(y_true_inverse, y_pred_inverse)

    # -----------------------------
    # Scatter plot
    # -----------------------------
    figsize = (2, 2)
    scale_factor = figsize[0]

    plt.figure(figsize=figsize)
    sns.regplot(
        x=y_true_inverse,
        y=y_pred_inverse,
        marker='o',
        scatter_kws={'color': '#005AB5', 's': 2.0, 'alpha': 0.6},
        line_kws={'color': '#0C7BDC'}
    )

    plt.xlabel("Measured AGB (Mg/ha)", fontsize=4 * scale_factor)
    plt.ylabel("Estimated AGB (Mg/ha)", fontsize=4 * scale_factor)

    # Find the maximum value
    max_value = max(max(y_true_inverse), max(y_pred_inverse))
    
    print("\nmax of y：",max(y_true_inverse))
    print("max of y_pred：",max(y_pred_inverse))
    print(f"max vlaue in {tipe}: {max_value}")

    # Set x-y axis limit from 0 to top
    top = max_value
    # Round 'top' up to the nearest multiple of 100
    rounded_top = math.ceil(top / 100.0) * 100
    plt.xlim(0, rounded_top)
    plt.ylim(0, rounded_top)

    # # Adjust the size of x and y axis tick labels
    plt.xticks(ticks=np.arange(0, rounded_top+1, 100),fontsize=4 * scale_factor, rotation=90)
    plt.yticks(ticks=np.arange(0, rounded_top+1, 100),fontsize=4 * scale_factor)
    
    # Plot the diagonal line representing perfect predictions
    plt.plot([0, rounded_top], [0, rounded_top], 
             color='black', 
             linestyle='--', 
             linewidth=0.7,
             alpha=0.7)

    # Equation of the regression line (slope and intercept)
    slope, intercept = np.polyfit(y_true_inverse.ravel(), y_pred_inverse.ravel(), 1)

    # Display the regression equation, R^2, and RMSE
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}\n" \
                    f"$R^2$ = {r2:.2f}\n" \
                    f"RMSE = {rmse:.2f}\n" \
                    f"rRMSE = {rrmse:.2f}%"

    plt.text(0.05, 0.95, 
             equation_text, 
             transform=plt.gca().transAxes,
             fontsize=3.5* scale_factor, 
             verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.0))

    # Create a DataFrame with y and y_pred
    results_df = pd.DataFrame({
        'Measured_Biomass': y_true_inverse.ravel(),
        'Predicted_Biomass': y_pred_inverse.ravel()
    })

    # Export the DataFrame to a CSV file
    results_df.to_csv(f"{save_dir}plot_scatter_{tipe}.csv", index=False)

    # Save the plot to a file
    plt.savefig(f"{save_dir}plot_scatter_{tipe}.png", dpi=600, pad_inches=0.02, bbox_inches='tight')
    # plt.gca().set_aspect('equal', adjustable='box') 
    
    # Show plot
    # plt.show(block=False)
    plt.close()
    print(f"\nScatter plot for {tipe}, prediction datafram exported successfully!")
    
    return


# history of optimization
def plot_optuna_results(study, save_dir, width=500, height=500):
    """
    Visualize the optimization history of an Optuna study.

    Parameters:
    study: Optuna study object containing the results of hyperparameter optimization
    save_dir (str): Directory to save output files
    width (int): Width of the plots (default: 500)
    height (int): Height of the plots (default: 500)

    Returns:
    None
    """
    
    # Plot optimization history
    fig_optimization_history = optuna.visualization.plot_optimization_history(study)
    fig_optimization_history.update_layout(
        title="Optimization history plot",
        xaxis_title="Step (Num of trial)",
        yaxis_title="Objective value (RMSE)",
        legend_title="Legend",
        # width=width,
        # height=height,
    )
    fig_optimization_history.write_html(f"{save_dir}plot_optimization_history.html")
    #fig_optimization_history.show(close=True)

    # Plot hyperparameter importance
    fig_param_importances = optuna.visualization.plot_param_importances(
        study,
        evaluator=MeanDecreaseImpurityImportanceEvaluator(),
    )
    fig_param_importances.update_layout(
        title="Hyperparameter importance",
        xaxis_title="SHAP Importance",
        yaxis_title="Hyperparameter",
        # width=width,
        # height=height,        
    )
    fig_param_importances.write_html(f"{save_dir}plot_param_importances.html")
    #fig_param_importances.show(close=True)

    # Compute hyperparameter importances and extract names ordered by importance
    importances = get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    param_names_by_importance = list(importances.keys())

    # Plot slice of hyperparameters
    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.update_layout(
        title="slice_plot",
        # xaxis_title="Objective value",
        # yaxis_title="Hyperparameter",
        # width=width,
        # height=height, 
    )
    fig_slice.write_html(f"{save_dir}plot_slice_plot.html")
    #fig_slice.show(close=True)

    print("\nTraining history saved successfully!")
    
    return


def compute_and_plot_shap(final_model, X_train_tensor, cols, timestamp, save_dir):
    """
    Compute and plot SHAP values with automatic GPU/CPU selection.
    
    Parameters:
    final_model: Trained model for SHAP explanation
    X_train_tensor: X_train_tensor
    save_dir (str): Directory to save the output files
    """
    import numpy as np
    import pandas as pd
    import torch
    import shap
    import matplotlib.pyplot as plt

    DEVICE = next(final_model.parameters()).device

    # -----------------------------
    # Model setup
    # -----------------------------
    old_cudnn = torch.backends.cudnn.enabled
    old_mode = final_model.training

    torch.backends.cudnn.enabled = False
    final_model.eval()

    # -----------------------------
    # Background samples
    # -----------------------------
    idx = torch.randperm(X_train_tensor.size(0))[:100]
    background = X_train_tensor[idx].to(DEVICE)

    # -----------------------------
    # SHAP computation
    # -----------------------------
    explainer = shap.GradientExplainer(final_model, background)
    shap_values = explainer.shap_values(X_train_tensor.to(DEVICE))

    shap_values = np.array(shap_values)
    shap_values = np.squeeze(shap_values)

    if shap_values.ndim != 3:
        raise ValueError(f"Expected SHAP shape (N, T, F), got {shap_values.shape}")

    N, T, F = shap_values.shape
    print(f"✅ SHAP shape: (samples={N}, timesteps={T}, features={F})")

    # Restore model state
    torch.backends.cudnn.enabled = old_cudnn
    final_model.train() if old_mode else final_model.eval()

    # Convert input data
    data_np = X_train_tensor.detach().cpu().numpy()
    
    # ============================================================
    #  timestamp setting
    # ============================================================
    mapping = {
        2: ["Semiannual1", "Semiannual2"],
        4: ["Q1", "Q2", "Q3", "Q4"],
        12: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    }
    
    if timestamp in mapping:
        timestep_names = mapping[timestamp]
        # This creates the list by iterating through timesteps first, then columns
        expanded_cols = [f"{c}_{timestep_names[t]}" for t in range(T) for c in cols]
    
    # ============================================================
    # 1️⃣ FEATURE IMPORTANCE (125 features)
    # Aggregate over time
    # ============================================================
    shap_feature = np.mean(shap_values, axis=1)     # (N, F)
    data_feature = np.mean(data_np, axis=1)         # (N, F)

    shap_exp_feature = shap.Explanation(
        values=shap_feature,
        data=data_feature,
        feature_names=cols
    )
    
    max_display_F = len(cols)
    
    plt.figure(figsize=(4, max(1, max_display_F * 0.3)))
    shap.plots.beeswarm(shap_exp_feature, max_display=max_display_F, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=20)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=20,labelpad=-40)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  
    # plt.title("Feature Importance (Aggregated over time)")
    plt.savefig(f"{save_dir}shap_beeswarm_feature.png", dpi=600, bbox_inches='tight')
    plt.close()

    # Save CSV (mean |SHAP|)
    feature_importance = np.mean(np.abs(shap_feature), axis=0)
    pd.DataFrame({
        "feature": cols,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)\
     .to_csv(f"{save_dir}shap_feature_importance.csv", index=False)

    print("✅ Feature beeswarm saved")

    # ============================================================
    # 2️⃣ TIME-STEP IMPORTANCE (2 steps)
    # Aggregate over features
    # ============================================================
    shap_time = np.mean(shap_values, axis=2)   # (N, T)
    data_time = np.mean(data_np, axis=2)       # (N, T)

    shap_exp_time = shap.Explanation(
        values=shap_time,
        data=data_time,
        feature_names=timestep_names
    )
    
    max_display_T = len(timestep_names)
    
    plt.figure(figsize=(4, max(1, max_display_T * 0.3)))
    shap.plots.beeswarm(shap_exp_time, max_display=max_display_T, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    # Make colorbar visibly thicker
    # cbar.set_aspect(20)                 
    cbar.set_box_aspect(20)
    cbar.tick_params(labelsize=15)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=15, labelpad=-20)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)      
    # plt.title("Time-step Importance")
    plt.savefig(f"{save_dir}shap_beeswarm_timestep.png", dpi=600, bbox_inches='tight')
    plt.close()

    # Save CSV
    time_importance = np.mean(np.abs(shap_time), axis=0)
    pd.DataFrame({
        "timestep": timestep_names,
        "importance": time_importance
    }).to_csv(f"{save_dir}shap_timestep_importance.csv", index=False)

    print("✅ Time-step beeswarm saved")

    # ============================================================
    # 3️⃣ OPTIONAL: Full 250-feature beeswarm (advanced)
    # ============================================================
    shap_flat = shap_values.reshape(N, T * F)
    data_flat = data_np.reshape(N, T * F)

    shap_exp_full = shap.Explanation(
        values=shap_flat,
        data=data_flat,
        feature_names=expanded_cols
    )

    max_display_Full =  min(125, len(expanded_cols))
    
    plt.figure(figsize=(4, max(1, max_display_Full * 0.3)))
    shap.plots.beeswarm(shap_exp_full, max_display=max_display_Full, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=20)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=20,labelpad=-40)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)   
    # plt.title("Full SHAP (Feature × Time)")
    plt.savefig(f"{save_dir}shap_beeswarm_full.png", dpi=600, bbox_inches='tight')
    plt.close()

    print("✅ Full beeswarm saved")

    return





# execution 
if __name__ == "__main__":

    def create_directory(base_path = "../run/ssp/bigru/before_boruta/month/"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()
   
    # define columns in model training
    cols =[
           'S1_RVI', 'S1_VH+VV', 'S1_VH-VV', 'S1_VH/VV', 
           'S1_VH', 
           'S1_VH_asm', 'S1_VH_con', 'S1_VH_corr', 'S1_VH_dent', 'S1_VH_diss', 'S1_VH_dvar', 'S1_VH_ent', 'S1_VH_homo', 'S1_VH_imcorr1', 'S1_VH_imcorr2', 
           'S1_VH_inertia', 'S1_VH_prom', 'S1_VH_savg', 'S1_VH_sent', 'S1_VH_shade', 'S1_VH_svar', 'S1_VH_var', 
           'S1_VV', 
           'S1_VV_asm', 'S1_VV_con', 'S1_VV_corr', 'S1_VV_dent', 'S1_VV_diss', 'S1_VV_dvar', 'S1_VV_ent', 'S1_VV_homo', 'S1_VV_imcorr1', 'S1_VV_imcorr2', 
           'S1_VV_inertia', 'S1_VV_prom', 'S1_VV_savg', 'S1_VV_sent', 'S1_VV_shade', 'S1_VV_svar', 'S1_VV_var', 
           'S2_B2', 'S2_B3', 'S2_B4', 'S2_B5', 'S2_B6', 'S2_B7', 'S2_B8', 'S2_B8A', 'S2_B11', 'S2_B12',
           'S2_CIgreen', 'S2_CIre', 'S2_DVI', 'S2_EVI1', 'S2_EVI2', 'S2_EVIre1', 'S2_EVIre2', 'S2_EVIre3', 'S2_GNDVI', 'S2_IRECI', 
           'S2_MCARI1', 'S2_MCARI2', 'S2_MCARI3', 'S2_MTCI1', 'S2_MTCI2', 'S2_MTCI3', 'S2_NDI45', 'S2_NDRE1', 'S2_NDRE2', 'S2_NDRE3', 
           'S2_NDVI56', 'S2_NDVI57', 'S2_NDVI68a', 'S2_NDVI78a', 'S2_NDWI1', 'S2_NDWI2', 'S2_NIRv', 'S2_NLI', 'S2_OSAVI', 'S2_PSSRa', 'S2_SAVI', 'S2_SR', 'S2_kNDVI', 
           'P2_HH+HV', 'P2_HH-HV', 
           'P2_HH', 
           'P2_HH_asm', 'P2_HH_con', 'P2_HH_corr', 'P2_HH_dent', 'P2_HH_diss', 'P2_HH_dvar', 'P2_HH_ent', 'P2_HH_homo', 'P2_HH_imcorr1', 'P2_HH_imcorr2', 
           'P2_HH_inertia', 'P2_HH_prom', 'P2_HH_savg', 'P2_HH_sent', 'P2_HH_shade', 'P2_HH_svar', 'P2_HH_var', 
           'P2_HV/HH', 'P2_HV', 
           'P2_HV_asm', 'P2_HV_con', 'P2_HV_corr', 'P2_HV_dent', 'P2_HV_diss', 'P2_HV_dvar', 'P2_HV_ent', 'P2_HV_homo', 'P2_HV_imcorr1', 'P2_HV_imcorr2', 
           'P2_HV_inertia', 'P2_HV_prom', 'P2_HV_savg', 'P2_HV_sent', 'P2_HV_shade', 'P2_HV_svar', 'P2_HV_var', 
           'Aspect', 'Ele', 'Slope'
          ]
    
    # load data: train, val and test
    for dataset in['training', 'validation', 'test']:
        if dataset == 'training':
            train_path = "../data/train_month_median.csv"  
            
            X_train, y_train = prepare_df(train_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_train, dataset, directory)
            X_train_scaled, y_train_scaled,  feature_scaler, target_scaler = standard_df(X_train, y_train,  directory)
            X_3d_scaled, y_2d_scaled = prepare_seq(X_train_scaled, y_train_scaled, timestamp=12)            
            # print(f"\nSample of first sequence (X_3d_scaled):")
            # print(X_3d_scaled[0])   
            
            # split cover
            X_train_Croplands, y_train_Croplands = prepare_df(train_path, cols, dataset, directory, cover_type="Croplands")
            X_train_Croplands_scaled = feature_scaler.transform(X_train_Croplands)
            y_train_Croplands_scaled = target_scaler.transform(y_train_Croplands.values.reshape(-1, 1))
            X_3d_Croplands_scaled, y_2d_Croplands_scaled = prepare_seq(X_train_Croplands_scaled, y_train_Croplands_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Croplands_scaled):")
            # print(X_3d_Croplands_scaled[0])    
            
            X_train_Forests, y_train_Forests = prepare_df(train_path, cols, dataset, directory, cover_type="Forests")
            X_train_Forests_scaled = feature_scaler.transform(X_train_Forests)
            y_train_Forests_scaled = target_scaler.transform(y_train_Forests.values.reshape(-1, 1))
            X_3d_Forests_scaled, y_2d_Forests_scaled = prepare_seq(X_train_Forests_scaled, y_train_Forests_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Forests_scaled):")
            # print(X_3d_Forests_scaled[0]) 
            
            X_train_Savannas, y_train_Savannas= prepare_df(train_path, cols, dataset, directory, cover_type="Savannas")
            X_train_Savannas_scaled = feature_scaler.transform(X_train_Savannas)
            y_train_Savannas_scaled = target_scaler.transform(y_train_Savannas.values.reshape(-1, 1))
            X_3d_Savannas_scaled, y_2d_Savannas_scaled = prepare_seq(X_train_Savannas_scaled, y_train_Savannas_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Savannas_scaled):")
            # print(X_3d_Savannas_scaled[0]) 
            
            X_train_Shrub_grass_lands, y_train_Shrub_grass_lands = prepare_df(train_path, cols, dataset, directory, cover_type="Shrub_grass_lands")
            X_train_Shrub_grass_lands_scaled = feature_scaler.transform(X_train_Shrub_grass_lands)
            y_train_Shrub_grass_lands_scaled = target_scaler.transform(y_train_Shrub_grass_lands.values.reshape(-1, 1))
            X_3d_Shrub_grass_lands_scaled, y_2d_Shrub_grass_lands_scaled = prepare_seq(X_train_Shrub_grass_lands_scaled, y_train_Shrub_grass_lands_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Shrub_grass_lands_scaled):")
            # print(X_3d_Shrub_grass_lands_scaled[0])             
       
        elif dataset == "validation":
            val_path ="../data/val_month_median.csv" 
            X_val, y_val = prepare_df(val_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_val, dataset, directory)
            X_val_scaled = feature_scaler.transform(X_val)
            y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
            X_3d_val_scaled, y_2d_val_scaled = prepare_seq(X_val_scaled, y_val_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_val_scaled):")
            # print(X_3d_val_scaled[0])
            
            
        else:
            test_path = "../data/test_month_median.csv"  
            
            X_test, y_test = prepare_df(test_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_test, dataset, directory)
            X_test_scaled = feature_scaler.transform(X_test)
            y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
            X_3d_test_scaled, y_2d_test_scaled = prepare_seq(X_test_scaled, y_test_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_test_scaled):")
            # print(X_3d_test_scaled[0])
            
            # split cover
            X_test_Croplands, y_test_Croplands = prepare_df(test_path, cols, dataset, directory, cover_type="Croplands")
            X_test_Croplands_scaled = feature_scaler.transform(X_test_Croplands)
            y_test_Croplands_scaled = target_scaler.transform(y_test_Croplands.values.reshape(-1, 1))
            X_3d_Croplands_scaled, y_2d_Croplands_scaled = prepare_seq(X_test_Croplands_scaled, y_test_Croplands_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Croplands_scaled):")
            # print(X_3d_Croplands_scaled[0])            
            
            X_test_Forests, y_test_Forests = prepare_df(test_path, cols, dataset, directory, cover_type="Forests")
            X_test_Forests_scaled = feature_scaler.transform(X_test_Forests)
            y_test_Forests_scaled = target_scaler.transform(y_test_Forests.values.reshape(-1, 1))
            X_3d_Forests_scaled, y_2d_Forests_scaled = prepare_seq(X_test_Forests_scaled, y_test_Forests_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Forests_scaled):")
            # print(X_3d_Forests_scaled[0])            
            
            X_test_Savannas, y_test_Savannas= prepare_df(test_path, cols, dataset, directory, cover_type="Savannas")
            X_test_Savannas_scaled = feature_scaler.transform(X_test_Savannas)
            y_test_Savannas_scaled = target_scaler.transform(y_test_Savannas.values.reshape(-1, 1))
            X_3d_Savannas_scaled, y_2d_Savannas_scaled = prepare_seq(X_test_Savannas_scaled, y_test_Savannas_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Savannas_scaled):")
            # print(X_3d_Savannas_scaled[0])            
            
            
            X_test_Shrub_grass_lands, y_test_Shrub_grass_lands = prepare_df(test_path, cols, dataset, directory, cover_type="Shrub_grass_lands")
            X_test_Shrub_grass_lands_scaled = feature_scaler.transform(X_test_Shrub_grass_lands)
            y_test_Shrub_grass_lands_scaled = target_scaler.transform(y_test_Shrub_grass_lands.values.reshape(-1, 1))  
            X_3d_Shrub_grass_lands_scaled, y_2d_Shrub_grass_lands_scaled= prepare_seq(X_test_Shrub_grass_lands_scaled, y_test_Shrub_grass_lands_scaled, timestamp=12)
            # print(f"\nSample of first sequence (X_3d_Shrub_grass_lands_scaled):")
            # print(X_3d_Shrub_grass_lands_scaled[0])          
            
            
    print(f"\n⚡X_train.head(1): \n{X_train.head(1)}")   

    # --- Configuration (Non-tunable) ---
    INPUT_SIZE = len(cols) # Number of input features
    OUTPUT_SIZE = 1  # Number of output neurons 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔋 Model was trained via {DEVICE}")

    # to tensor
    train_tensor = prepare_tensor(X_3d_scaled, y_2d_scaled)
    X_train_tensor, y_train_tensor = train_tensor.tensors  # tensors is a tuple (X, y)
    print("\n⚡X_train_tensor shape:", X_train_tensor.shape)
    print(f'X_train_tensor[:1]: \n{X_train_tensor[:1]}')
    train_tensor_Croplands = prepare_tensor(X_3d_Croplands_scaled, y_2d_Croplands_scaled)
    train_tensor_Forests = prepare_tensor(X_3d_Forests_scaled, y_2d_Forests_scaled)
    train_tensor_Savannas = prepare_tensor(X_3d_Savannas_scaled, y_2d_Savannas_scaled)
    train_tensor_Shrub_grass_lands = prepare_tensor(X_3d_Shrub_grass_lands_scaled, y_2d_Shrub_grass_lands_scaled)
    
    # val_scaled to tensor        
    val_tensor = prepare_tensor(X_3d_val_scaled, y_2d_val_scaled)
    
    # test_scaled to tensor  
    test_tensor = prepare_tensor(X_3d_test_scaled, y_2d_test_scaled)
    test_tensor_Croplands = prepare_tensor(X_3d_Croplands_scaled, y_2d_Croplands_scaled)
    test_tensor_Forests = prepare_tensor(X_3d_Forests_scaled, y_2d_Forests_scaled)
    test_tensor_Savannas = prepare_tensor(X_3d_Savannas_scaled, y_2d_Savannas_scaled)
    test_tensor_Shrub_grass_lands = prepare_tensor(X_3d_Shrub_grass_lands_scaled, y_2d_Shrub_grass_lands_scaled)
    # training starts
    starttime = datetime.now()
    
    # --- 5. Run Optimization ---
    # Set up the Optuna study using the TPESampler and 
    study = optuna.create_study(
            # sampler=optuna.samplers.RandomSampler(seed=42),
            sampler=optuna.samplers.TPESampler(n_startup_trials=100),
            # sampler=optuna.samplers.GPSampler(n_startup_trials=100),,
            direction='minimize'
        )
    # optimization
    from functools import partial
    study.optimize(partial(objective, save_dir=directory), 
                   n_trials=2
                  )
    
    print("\nOptimization Complete.")
    print(f"Best Value (MSE): {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)

    #end_time
    endtime = datetime.now()
    used_time = endtime - starttime
    
    print(f"\n--xgb training time--: {used_time }")


    # retrain the model
    final_model = retrain(study, INPUT_SIZE, OUTPUT_SIZE, train_tensor, val_tensor, directory)
        
    # save train and test scatter plot
    for tipe in ['train_all','train_crop','train_forest','train_savannas','train_shrubgrass',
                 'test_all','test_crop','test_forest','test_savannas','test_shrubgrass']:
        if tipe == 'train_all':
            data_tensor = train_tensor
        elif tipe == 'train_crop':
            data_tensor = train_tensor_Croplands
        elif tipe == 'train_forest':
            data_tensor = train_tensor_Forests
        elif tipe == 'train_savannas':
            data_tensor = train_tensor_Savannas
        elif tipe == 'train_shrubgrass':
            data_tensor = train_tensor_Shrub_grass_lands
        elif tipe == 'test_all':
            data_tensor = test_tensor
        elif tipe == 'test_crop':
            data_tensor = test_tensor_Croplands
        elif tipe == 'test_forest':
            data_tensor = test_tensor_Forests
        elif tipe == 'test_savannas':
            data_tensor = test_tensor_Savannas
        else:
            data_tensor = test_tensor_Shrub_grass_lands   
            
        evaluate_and_plot(final_model, data_tensor, feature_scaler, target_scaler, tipe, directory)
    
    # optuna history
    plot_optuna_results(study, directory)
    
    # compute SHAP
    compute_and_plot_shap(final_model, X_train_tensor, cols, timestamp=12, save_dir=directory)
    print("\nExecution done!")         