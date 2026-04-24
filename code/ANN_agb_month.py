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


import math
import optuna
print("\n******Current working dir", os.getcwd())

print('torch=:',torch.__version__)
print('optuna=:',optuna.__version__)
# print('optunahub=:',optunahub.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# load and preprocess data
def prepare_df(csv_path, cols, dataset, save_dir, cover_type=None):
    """
    Load data from a CSV file, preprocess it, and convert it into a df for rf.

    Parameters:
    - csv_path: Path to the CSV file.
    - cols: List of column names for features.
    - dataset: train / validation / test
    - save_dir: Directory path to save preprocessed data.
    - cover_type: landcover type to filter (optional)

    Returns:
    - X: DataFrame of features.
    - y: Series of target variable.
    """

    # Load the data
    df = pd.read_csv(csv_path)

    # Filter by land cover if specified
    if cover_type is not None:
        df = df[df["Cover"] == cover_type]
        print(f"Filtering for cover type: {cover_type}")
    else:
        print("Using all land cover types")

    # Select features and target
    X = df[cols]
    y = df["AGBD"]

    print(f"\nTotal {dataset} samples:", len(X))

    print("\nRF dataframe prepared successfully!")

    with open(f"{save_dir}_used_features.txt", "w") as file:
        file.write(f"used features:\n{cols}")

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

def standard_df(X_train, y_train, save_dir):
    """
    Load data from a CSV file, preprocess it, and convert it into a df for nn.

    Parameters:
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X_train: DataFrame of features.
    - y_train: Series of target variable.
    """

    # Scale features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Scale features (X is already 2D)
    X_train_scaled = feature_scaler.fit_transform(X_train)

    # Scale target (must be 2D)
    y_train_scaled = target_scaler.fit_transform(
        y_train.values.reshape(-1, 1)
    )

    # save scaler
    joblib.dump(feature_scaler, f'{save_dir}feature_scaler.joblib')
    joblib.dump(target_scaler, f'{save_dir}target_scaler.joblib')

                       
    print(f"\n🚀X_train.head(1):\n", X_train.head(1))
    print(f"\n🚀X_train_scaled[:1]:\n", X_train_scaled[:1])
    print(f"\n🚀y_train.head(1):\n", y_train.head(1))
    print(f"\n🚀y_train_scaled[:1]:\n", y_train_scaled[:1])
    print("\nDataframe StandardScaler successfully!")
    
    return X_train_scaled, y_train_scaled,  feature_scaler, target_scaler


def prepare_tensor(X_scaled: np.ndarray, y_scaled: np.ndarray):
    """
    Convert scaled data into a PyTorch TensorDataset.
    Input shape: X (n_samples, 64), y (n_samples, 1)
    """
    X_tensor = torch.FloatTensor(X_scaled)          # (n_samples, 64)
    y_tensor = torch.FloatTensor(y_scaled)          # (n_samples, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    print(f"✅ TensorDataset created → {len(dataset)} samples, "
          f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    return dataset

# # -----------------------------
# # Model Definition (residual connections)
# #  initialization kaiming_uniform_(a=√5)
# # -----------------------------
# class MLPBlock(nn.Module):
#     def __init__(self, in_features, out_features, dropout, use_batchnorm=False):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
        
#         if use_batchnorm:
#             self.norm = nn.BatchNorm1d(out_features)
#         else:
#             self.norm = nn.LayerNorm(out_features)
            
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout)
        
#         # Add projection for residual connection if dimensions don't match
#         self.proj = None
#         if in_features != out_features:
#             self.proj = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         residual = x
        
#         x = self.linear(x)
#         x = self.norm(x)
#         x = self.relu(x)
#         x = self.dropout(x)
        
#         # Apply projection to residual if needed (for dimension matching)
#         if self.proj is not None:
#             residual = self.proj(residual)
        
#         # Residual addition
#         x = x + residual
#         return x

# class MLPRegressor(nn.Module):
#     def __init__(self, input_size, output_size, hidden_sizes, dropouts, use_batchnorm=False):
#         super().__init__()
#         if len(hidden_sizes) != len(dropouts):
#             raise ValueError("hidden_sizes and dropouts must have the same length")
#         layers = []
#         prev_size = input_size
        
#         for h_size, d_prob in zip(hidden_sizes, dropouts):
#             layers.append(MLPBlock(prev_size, h_size, d_prob, use_batchnorm))
#             prev_size = h_size
            
#         self.hidden_layers = nn.Sequential(*layers)
#         self.output_layer = nn.Linear(prev_size, output_size)
        
#     def forward(self, x):
#         x = self.hidden_layers(x)
#         return self.output_layer(x)



# -----------------------------
# Model Definition (residual connections)
#  initialization original kaiming_uniform
# -----------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, use_batchnorm=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # === Proper He/Kaiming for ReLU ===
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)          # best practice instead of uniform bias

        if use_batchnorm:  
            self.norm = nn.BatchNorm1d(out_features)  
        else:  
            self.norm = nn.LayerNorm(out_features)  
              
        self.relu = nn.ReLU(inplace=True)  
        self.dropout = nn.Dropout(dropout)  
          
        # Residual projection (also needs correct init)
        self.proj = None
        if in_features != out_features:  
            self.proj = nn.Linear(in_features, out_features)
            nn.init.kaiming_uniform_(self.proj.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):  
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.proj is not None:
            residual = self.proj(residual)
        
        x = x + residual
        return x


class MLPRegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropouts, use_batchnorm=False):
        super().__init__()
        if len(hidden_sizes) != len(dropouts):
            raise ValueError("hidden_sizes and dropouts must have the same length")
        
        layers = []
        prev_size = input_size

        for h_size, d_prob in zip(hidden_sizes, dropouts):  
            layers.append(MLPBlock(prev_size, h_size, d_prob, use_batchnorm))  
            prev_size = h_size
              
        self.hidden_layers = nn.Sequential(*layers)  
        
        # Final regression head — also He init (though linear output is less critical)
        self.output_layer = nn.Linear(prev_size, output_size)
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)
      
    def forward(self, x):  
        x = self.hidden_layers(x)
        return self.output_layer(x)





# -----------------------------
# Optuna Objective
# -----------------------------
def objective(trial, save_dir):
    
    # --- 1. Define Search Space ---
    num_layers = trial.suggest_int("num_layers", 1, 16, step=1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=False)  # log scale often better
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0])  # log scale often better
    initial_lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 1e-5])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

    hidden_sizes = []
    dropouts = []
    for i in range(num_layers):
        layer_width = trial.suggest_categorical(f"layer_{i}_size", [16, 32, 64, 128, 256, 512])
        hidden_sizes.append(layer_width)
        
        layer_dropout = trial.suggest_float(f"layer_{i}_dropout", 0.0, 0.6, step=0.1)
        dropouts.append(layer_dropout)
    
    # Conditional normalization based on batch_size
    use_batchnorm = True if batch_size >= 32 else False
    # logging/printing
    norm_type = "BatchNorm1d" if use_batchnorm else "LayerNorm"
    print(f"\n🚩Trial {trial.number} | batch_size={batch_size} → Using {norm_type}")
    print(f"Architecture: {hidden_sizes} | Dropouts: {dropouts}")

    # --- 2. DataLoaders ---
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    # --- 3. Model, Optimizer, Scheduler, Loss ---
    model = MLPRegressor(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        use_batchnorm=use_batchnorm
    ).to(DEVICE)
    
    # --- 4. Optimizer Initialization ---
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("rmsprop_momentum", 0.85, 0.99, step=0.01)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.85, 0.99, step=0.01)
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
        min_lr=0, #1e-6,
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
    
    # Reconstruct architecture lists
    hidden_sizes = [best_params[f"layer_{i}_size"] for i in range(num_layers)]
    dropouts = [best_params[f"layer_{i}_dropout"] for i in range(num_layers)]
    
    use_batchnorm = True if batch_size >= 32 else False
    
    print("\n" + "="*30)
    print(f"🚀 RETRAINING BEST MODEL (Trial {study.best_trial.number})")
    print(f"Optimizer: {optimizer_name} | Batch Size: {batch_size}")
    print(f"Architecture: {hidden_sizes}")
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
    model = MLPRegressor(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        use_batchnorm=use_batchnorm
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
        min_lr=0#1e-6,
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

    best_model_path = f"{save_dir}/_final_model.pth"

    # -----------------------------
    # Loss tracking
    # -----------------------------
    epoch_list = []
    train_mse_list = []
    val_mse_list = []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(EPOCHS):
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

        # Early stopping logic
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

        scheduler.step(avg_val_mse)

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
    plt.xticks(ticks=np.arange(0, rounded_top+1, 10),fontsize=4 * scale_factor, rotation=0)
    
    plt.xlabel("Epoch", fontsize=4* scale_factor)
    plt.ylabel("MSE", fontsize=4* scale_factor)
    # plt.title("Training & Validation Loss", fontsize=4* scale_factor)
    plt.legend(frameon=False, fontsize=3* scale_factor)
    # plt.grid(True)

    plot_path = f"{save_dir}/re_training_val_loss.png"
    plt.savefig(plot_path, dpi=300, pad_inches=0.02, bbox_inches='tight')
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
        shuffle=False
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

# SHAP based feature importance 
def compute_and_plot_shap(final_model, X_train_tensor, cols, save_dir):
    """
    Compute and plot SHAP values with automatic GPU/CPU selection.
    
    Parameters:
    final_model: Trained model for SHAP explanation
    X_train_tensor: X_train_tensor
    save_dir (str): Directory to save the output files
    """
    # random select certain samples in X_train_tensor as background
    idx = torch.randperm(X_train_tensor.size(0))[:100]
    background = X_train_tensor[idx].to(DEVICE)
    final_model.eval()
    explainer = shap.GradientExplainer(final_model,background)
   
    # Compute SHAP values
    # How much did each input feature contribute to the model output?
    shap_values_array = explainer.shap_values(X_train_tensor.to(DEVICE))
    print(f"🚀shap_values_array.shape: {shap_values_array.shape}")
    print(shap_values_array)
    
    # Decide which slice to use for 2D plots (bar / violin / beeswarm)
    if isinstance(shap_values_array, list):
        # multi-output → pick one (e.g. class 0, or the positive class, etc.)
        shap_2d = shap_values_array[0]           # ← adjust index as needed
    elif len(shap_values_array.shape) == 3:
        # shape = (samples, features, outputs) → pick one output
        shap_2d = shap_values_array[:, :, 0]     # or .mean(axis=2), etc.
    else:
        # already 2D: (samples, features)
        shap_2d = shap_values_array
    print(f"🚀shap_2d.shape: {shap_2d.shape}")
    print(shap_2d)

    # This is the original input data, converted to NumPy
    data_np = X_train_tensor.detach().cpu().numpy()
    print(f"🚀data_np.shape: {data_np.shape}")
    print(data_np)
    
    # object: Rich wrapper that bundles SHAP values (shap_2d), original data (data_np), feature names
    # Modern SHAP plotting API expects this object: shap.plots.bar(), .beeswarm(), .violin() etc.
    shap_values = shap.Explanation(
        values=shap_2d,
        data=data_np,
        feature_names=cols)
    print(f"🚀shap_values.shape: {shap_values.shape}")
    print(shap_values)
    
    # Convert SHAP values to a DataFrame for saving
    shap_df = pd.DataFrame(shap_2d, columns=cols)
    shap_df.to_csv(f"{save_dir}shap_values.csv", index=False)
    print("\nSHAP values saved to CSV successfully!")

    # Set the max_display based on feature count
    max_display = min(64, X_train_tensor.shape[1])

    # Create and save the SHAP bar plot
    fig_bar = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.xlabel("Mean |SHAP value|", fontsize=23)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)
    fig_bar.savefig(f"{save_dir}shap_bar.png", dpi=600, bbox_inches='tight')
    plt.close(fig_bar)

    ###
    ### In modern SHAP: Beeswarm = summary plot
    ### There is no need for summary_plot
    # # Create and save the SHAP summary plot
    # fig_summary_plot = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    # shap.summary_plot(shap_values, max_display=max_display, show=False)
    # # Get the colorbar's axis object
    # cbar = plt.gcf().axes[-1]
    # cbar.tick_params(labelsize=23)
    # cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # # Set the label for the colorbar with a larger font size
    # cbar.set_ylabel("Feature value", fontsize=23,labelpad=-40)
    # plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    # plt.xticks(fontsize=21)
    # plt.yticks(fontsize=22)    
    # fig_summary_plot.savefig(f"{save_dir}shap_summary_plot.png", dpi=600, bbox_inches='tight')
    # plt.close(fig_summary_plot)

    # Create and save the SHAP violin plot
    fig_violin = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.violin(shap_values, max_display=max_display, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=23)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=23,labelpad=-40)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)    
    fig_violin.savefig(f"{save_dir}shap_violin.png", dpi=600, bbox_inches='tight')
    plt.close(fig_violin)

    # Create and save the SHAP beeswarm plot
    fig_beeswarm = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=23)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=23,labelpad=-40)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)    
    fig_beeswarm.savefig(f"{save_dir}shap_beeswarm.png", dpi=600, bbox_inches='tight')
    plt.close(fig_beeswarm)

    print("\nSHAP plots saved successfully!")
    
    return
    

# execution 1 time
if __name__ == "__main__":

    def create_directory(base_path = "../run/ssp/mlp/before_boruta/month/"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()

    # define columns in model training
    cols = [ 'Aspect', 'Ele', 'Slope',
             'S1_RVI_Jan', 'S1_RVI_Feb', 'S1_RVI_Mar', 'S1_RVI_Apr', 'S1_RVI_May', 'S1_RVI_Jun', 'S1_RVI_Jul', 'S1_RVI_Aug', 'S1_RVI_Sep', 'S1_RVI_Oct', 'S1_RVI_Nov', 'S1_RVI_Dec', 
             'S1_VH+VV_Jan', 'S1_VH+VV_Feb', 'S1_VH+VV_Mar', 'S1_VH+VV_Apr', 'S1_VH+VV_May', 'S1_VH+VV_Jun', 'S1_VH+VV_Jul', 'S1_VH+VV_Aug', 'S1_VH+VV_Sep', 'S1_VH+VV_Oct', 'S1_VH+VV_Nov', 'S1_VH+VV_Dec', 
             'S1_VH-VV_Jan', 'S1_VH-VV_Feb', 'S1_VH-VV_Mar', 'S1_VH-VV_Apr', 'S1_VH-VV_May', 'S1_VH-VV_Jun', 'S1_VH-VV_Jul', 'S1_VH-VV_Aug', 'S1_VH-VV_Sep', 'S1_VH-VV_Oct', 'S1_VH-VV_Nov', 'S1_VH-VV_Dec', 
             'S1_VH/VV_Jan', 'S1_VH/VV_Feb', 'S1_VH/VV_Mar', 'S1_VH/VV_Apr', 'S1_VH/VV_May', 'S1_VH/VV_Jun', 'S1_VH/VV_Jul', 'S1_VH/VV_Aug', 'S1_VH/VV_Sep', 'S1_VH/VV_Oct', 'S1_VH/VV_Nov', 'S1_VH/VV_Dec',
             'S1_VH_Jan', 'S1_VH_Feb', 'S1_VH_Mar', 'S1_VH_Apr', 'S1_VH_May', 'S1_VH_Jun', 'S1_VH_Jul', 'S1_VH_Aug', 'S1_VH_Sep', 'S1_VH_Oct', 'S1_VH_Nov', 'S1_VH_Dec', 
             'S1_VH_asm_Jan', 'S1_VH_asm_Feb', 'S1_VH_asm_Mar', 'S1_VH_asm_Apr', 'S1_VH_asm_May', 'S1_VH_asm_Jun', 'S1_VH_asm_Jul', 'S1_VH_asm_Aug', 'S1_VH_asm_Sep', 'S1_VH_asm_Oct', 'S1_VH_asm_Nov', 'S1_VH_asm_Dec', 
             'S1_VH_con_Jan', 'S1_VH_con_Feb', 'S1_VH_con_Mar', 'S1_VH_con_Apr', 'S1_VH_con_May', 'S1_VH_con_Jun', 'S1_VH_con_Jul', 'S1_VH_con_Aug', 'S1_VH_con_Sep', 'S1_VH_con_Oct', 'S1_VH_con_Nov', 'S1_VH_con_Dec',
             'S1_VH_corr_Jan', 'S1_VH_corr_Feb', 'S1_VH_corr_Mar', 'S1_VH_corr_Apr', 'S1_VH_corr_May', 'S1_VH_corr_Jun', 'S1_VH_corr_Jul', 'S1_VH_corr_Aug', 'S1_VH_corr_Sep', 'S1_VH_corr_Oct', 'S1_VH_corr_Nov', 'S1_VH_corr_Dec', 
             'S1_VH_dent_Jan', 'S1_VH_dent_Feb', 'S1_VH_dent_Mar', 'S1_VH_dent_Apr', 'S1_VH_dent_May', 'S1_VH_dent_Jun', 'S1_VH_dent_Jul', 'S1_VH_dent_Aug', 'S1_VH_dent_Sep', 'S1_VH_dent_Oct', 'S1_VH_dent_Nov', 'S1_VH_dent_Dec', 
             'S1_VH_diss_Jan', 'S1_VH_diss_Feb', 'S1_VH_diss_Mar', 'S1_VH_diss_Apr', 'S1_VH_diss_May', 'S1_VH_diss_Jun', 'S1_VH_diss_Jul', 'S1_VH_diss_Aug', 'S1_VH_diss_Sep', 'S1_VH_diss_Oct', 'S1_VH_diss_Nov', 'S1_VH_diss_Dec',
             'S1_VH_dvar_Jan', 'S1_VH_dvar_Feb', 'S1_VH_dvar_Mar', 'S1_VH_dvar_Apr', 'S1_VH_dvar_May', 'S1_VH_dvar_Jun', 'S1_VH_dvar_Jul', 'S1_VH_dvar_Aug', 'S1_VH_dvar_Sep', 'S1_VH_dvar_Oct', 'S1_VH_dvar_Nov', 'S1_VH_dvar_Dec',
             'S1_VH_ent_Jan', 'S1_VH_ent_Feb', 'S1_VH_ent_Mar', 'S1_VH_ent_Apr', 'S1_VH_ent_May', 'S1_VH_ent_Jun', 'S1_VH_ent_Jul', 'S1_VH_ent_Aug', 'S1_VH_ent_Sep', 'S1_VH_ent_Oct', 'S1_VH_ent_Nov', 'S1_VH_ent_Dec',
             'S1_VH_homo_Jan', 'S1_VH_homo_Feb', 'S1_VH_homo_Mar', 'S1_VH_homo_Apr', 'S1_VH_homo_May', 'S1_VH_homo_Jun', 'S1_VH_homo_Jul', 'S1_VH_homo_Aug', 'S1_VH_homo_Sep', 'S1_VH_homo_Oct', 'S1_VH_homo_Nov', 'S1_VH_homo_Dec', 
             'S1_VH_imcorr1_Jan', 'S1_VH_imcorr1_Feb', 'S1_VH_imcorr1_Mar', 'S1_VH_imcorr1_Apr', 'S1_VH_imcorr1_May', 'S1_VH_imcorr1_Jun', 'S1_VH_imcorr1_Jul', 'S1_VH_imcorr1_Aug', 'S1_VH_imcorr1_Sep', 'S1_VH_imcorr1_Oct', 'S1_VH_imcorr1_Nov', 'S1_VH_imcorr1_Dec', 
             'S1_VH_imcorr2_Jan', 'S1_VH_imcorr2_Feb', 'S1_VH_imcorr2_Mar', 'S1_VH_imcorr2_Apr', 'S1_VH_imcorr2_May', 'S1_VH_imcorr2_Jun', 'S1_VH_imcorr2_Jul', 'S1_VH_imcorr2_Aug', 'S1_VH_imcorr2_Sep', 'S1_VH_imcorr2_Oct', 'S1_VH_imcorr2_Nov', 'S1_VH_imcorr2_Dec',
             'S1_VH_inertia_Jan', 'S1_VH_inertia_Feb', 'S1_VH_inertia_Mar', 'S1_VH_inertia_Apr', 'S1_VH_inertia_May', 'S1_VH_inertia_Jun', 'S1_VH_inertia_Jul', 'S1_VH_inertia_Aug', 'S1_VH_inertia_Sep', 'S1_VH_inertia_Oct', 'S1_VH_inertia_Nov', 'S1_VH_inertia_Dec', 
             'S1_VH_prom_Jan', 'S1_VH_prom_Feb', 'S1_VH_prom_Mar', 'S1_VH_prom_Apr', 'S1_VH_prom_May', 'S1_VH_prom_Jun', 'S1_VH_prom_Jul', 'S1_VH_prom_Aug', 'S1_VH_prom_Sep', 'S1_VH_prom_Oct', 'S1_VH_prom_Nov', 'S1_VH_prom_Dec',
             'S1_VH_savg_Jan', 'S1_VH_savg_Feb', 'S1_VH_savg_Mar', 'S1_VH_savg_Apr', 'S1_VH_savg_May', 'S1_VH_savg_Jun', 'S1_VH_savg_Jul', 'S1_VH_savg_Aug', 'S1_VH_savg_Sep', 'S1_VH_savg_Oct', 'S1_VH_savg_Nov', 'S1_VH_savg_Dec', 
             'S1_VH_sent_Jan', 'S1_VH_sent_Feb', 'S1_VH_sent_Mar', 'S1_VH_sent_Apr', 'S1_VH_sent_May', 'S1_VH_sent_Jun', 'S1_VH_sent_Jul', 'S1_VH_sent_Aug', 'S1_VH_sent_Sep', 'S1_VH_sent_Oct', 'S1_VH_sent_Nov', 'S1_VH_sent_Dec', 
             'S1_VH_shade_Jan', 'S1_VH_shade_Feb', 'S1_VH_shade_Mar', 'S1_VH_shade_Apr', 'S1_VH_shade_May', 'S1_VH_shade_Jun', 'S1_VH_shade_Jul', 'S1_VH_shade_Aug', 'S1_VH_shade_Sep', 'S1_VH_shade_Oct', 'S1_VH_shade_Nov', 'S1_VH_shade_Dec', 
             'S1_VH_svar_Jan', 'S1_VH_svar_Feb', 'S1_VH_svar_Mar', 'S1_VH_svar_Apr', 'S1_VH_svar_May', 'S1_VH_svar_Jun', 'S1_VH_svar_Jul', 'S1_VH_svar_Aug', 'S1_VH_svar_Sep', 'S1_VH_svar_Oct', 'S1_VH_svar_Nov', 'S1_VH_svar_Dec',
             'S1_VH_var_Jan', 'S1_VH_var_Feb', 'S1_VH_var_Mar', 'S1_VH_var_Apr', 'S1_VH_var_May', 'S1_VH_var_Jun', 'S1_VH_var_Jul', 'S1_VH_var_Aug', 'S1_VH_var_Sep', 'S1_VH_var_Oct', 'S1_VH_var_Nov', 'S1_VH_var_Dec', 
             'S1_VV_Jan', 'S1_VV_Feb', 'S1_VV_Mar', 'S1_VV_Apr', 'S1_VV_May', 'S1_VV_Jun', 'S1_VV_Jul', 'S1_VV_Aug', 'S1_VV_Sep', 'S1_VV_Oct', 'S1_VV_Nov', 'S1_VV_Dec', 
             'S1_VV_asm_Jan', 'S1_VV_asm_Feb', 'S1_VV_asm_Mar', 'S1_VV_asm_Apr', 'S1_VV_asm_May', 'S1_VV_asm_Jun', 'S1_VV_asm_Jul', 'S1_VV_asm_Aug', 'S1_VV_asm_Sep', 'S1_VV_asm_Oct', 'S1_VV_asm_Nov', 'S1_VV_asm_Dec',
             'S1_VV_con_Jan', 'S1_VV_con_Feb', 'S1_VV_con_Mar', 'S1_VV_con_Apr', 'S1_VV_con_May', 'S1_VV_con_Jun', 'S1_VV_con_Jul', 'S1_VV_con_Aug', 'S1_VV_con_Sep', 'S1_VV_con_Oct', 'S1_VV_con_Nov', 'S1_VV_con_Dec',
             'S1_VV_corr_Jan', 'S1_VV_corr_Feb', 'S1_VV_corr_Mar', 'S1_VV_corr_Apr', 'S1_VV_corr_May', 'S1_VV_corr_Jun', 'S1_VV_corr_Jul', 'S1_VV_corr_Aug', 'S1_VV_corr_Sep', 'S1_VV_corr_Oct', 'S1_VV_corr_Nov', 'S1_VV_corr_Dec', 
             'S1_VV_dent_Jan', 'S1_VV_dent_Feb', 'S1_VV_dent_Mar', 'S1_VV_dent_Apr', 'S1_VV_dent_May', 'S1_VV_dent_Jun', 'S1_VV_dent_Jul', 'S1_VV_dent_Aug', 'S1_VV_dent_Sep', 'S1_VV_dent_Oct', 'S1_VV_dent_Nov', 'S1_VV_dent_Dec', 
             'S1_VV_diss_Jan', 'S1_VV_diss_Feb', 'S1_VV_diss_Mar', 'S1_VV_diss_Apr', 'S1_VV_diss_May', 'S1_VV_diss_Jun', 'S1_VV_diss_Jul', 'S1_VV_diss_Aug', 'S1_VV_diss_Sep', 'S1_VV_diss_Oct', 'S1_VV_diss_Nov', 'S1_VV_diss_Dec', 
             'S1_VV_dvar_Jan', 'S1_VV_dvar_Feb', 'S1_VV_dvar_Mar', 'S1_VV_dvar_Apr', 'S1_VV_dvar_May', 'S1_VV_dvar_Jun', 'S1_VV_dvar_Jul', 'S1_VV_dvar_Aug', 'S1_VV_dvar_Sep', 'S1_VV_dvar_Oct', 'S1_VV_dvar_Nov', 'S1_VV_dvar_Dec',
             'S1_VV_ent_Jan', 'S1_VV_ent_Feb', 'S1_VV_ent_Mar', 'S1_VV_ent_Apr', 'S1_VV_ent_May', 'S1_VV_ent_Jun', 'S1_VV_ent_Jul', 'S1_VV_ent_Aug', 'S1_VV_ent_Sep', 'S1_VV_ent_Oct', 'S1_VV_ent_Nov', 'S1_VV_ent_Dec', 
             'S1_VV_homo_Jan', 'S1_VV_homo_Feb', 'S1_VV_homo_Mar', 'S1_VV_homo_Apr', 'S1_VV_homo_May', 'S1_VV_homo_Jun', 'S1_VV_homo_Jul', 'S1_VV_homo_Aug', 'S1_VV_homo_Sep', 'S1_VV_homo_Oct', 'S1_VV_homo_Nov', 'S1_VV_homo_Dec', 
             'S1_VV_imcorr1_Jan', 'S1_VV_imcorr1_Feb', 'S1_VV_imcorr1_Mar', 'S1_VV_imcorr1_Apr', 'S1_VV_imcorr1_May', 'S1_VV_imcorr1_Jun', 'S1_VV_imcorr1_Jul', 'S1_VV_imcorr1_Aug', 'S1_VV_imcorr1_Sep', 'S1_VV_imcorr1_Oct', 'S1_VV_imcorr1_Nov', 'S1_VV_imcorr1_Dec', 
             'S1_VV_imcorr2_Jan', 'S1_VV_imcorr2_Feb', 'S1_VV_imcorr2_Mar', 'S1_VV_imcorr2_Apr', 'S1_VV_imcorr2_May', 'S1_VV_imcorr2_Jun', 'S1_VV_imcorr2_Jul', 'S1_VV_imcorr2_Aug', 'S1_VV_imcorr2_Sep', 'S1_VV_imcorr2_Oct', 'S1_VV_imcorr2_Nov', 'S1_VV_imcorr2_Dec',
             'S1_VV_inertia_Jan', 'S1_VV_inertia_Feb', 'S1_VV_inertia_Mar', 'S1_VV_inertia_Apr', 'S1_VV_inertia_May', 'S1_VV_inertia_Jun', 'S1_VV_inertia_Jul', 'S1_VV_inertia_Aug', 'S1_VV_inertia_Sep', 'S1_VV_inertia_Oct', 'S1_VV_inertia_Nov', 'S1_VV_inertia_Dec', 
             'S1_VV_prom_Jan', 'S1_VV_prom_Feb', 'S1_VV_prom_Mar', 'S1_VV_prom_Apr', 'S1_VV_prom_May', 'S1_VV_prom_Jun', 'S1_VV_prom_Jul', 'S1_VV_prom_Aug', 'S1_VV_prom_Sep', 'S1_VV_prom_Oct', 'S1_VV_prom_Nov', 'S1_VV_prom_Dec', 
             'S1_VV_savg_Jan', 'S1_VV_savg_Feb', 'S1_VV_savg_Mar', 'S1_VV_savg_Apr', 'S1_VV_savg_May', 'S1_VV_savg_Jun', 'S1_VV_savg_Jul', 'S1_VV_savg_Aug', 'S1_VV_savg_Sep', 'S1_VV_savg_Oct', 'S1_VV_savg_Nov', 'S1_VV_savg_Dec', 
             'S1_VV_sent_Jan', 'S1_VV_sent_Feb', 'S1_VV_sent_Mar', 'S1_VV_sent_Apr', 'S1_VV_sent_May', 'S1_VV_sent_Jun', 'S1_VV_sent_Jul', 'S1_VV_sent_Aug', 'S1_VV_sent_Sep', 'S1_VV_sent_Oct', 'S1_VV_sent_Nov', 'S1_VV_sent_Dec',
             'S1_VV_shade_Jan', 'S1_VV_shade_Feb', 'S1_VV_shade_Mar', 'S1_VV_shade_Apr', 'S1_VV_shade_May', 'S1_VV_shade_Jun', 'S1_VV_shade_Jul', 'S1_VV_shade_Aug', 'S1_VV_shade_Sep', 'S1_VV_shade_Oct', 'S1_VV_shade_Nov', 'S1_VV_shade_Dec',
             'S1_VV_svar_Jan', 'S1_VV_svar_Feb', 'S1_VV_svar_Mar', 'S1_VV_svar_Apr', 'S1_VV_svar_May', 'S1_VV_svar_Jun', 'S1_VV_svar_Jul', 'S1_VV_svar_Aug', 'S1_VV_svar_Sep', 'S1_VV_svar_Oct', 'S1_VV_svar_Nov', 'S1_VV_svar_Dec', 
             'S1_VV_var_Jan', 'S1_VV_var_Feb', 'S1_VV_var_Mar', 'S1_VV_var_Apr', 'S1_VV_var_May', 'S1_VV_var_Jun', 'S1_VV_var_Jul', 'S1_VV_var_Aug', 'S1_VV_var_Sep', 'S1_VV_var_Oct', 'S1_VV_var_Nov', 'S1_VV_var_Dec', 
             
             'S2_B2_Jan', 'S2_B2_Feb', 'S2_B2_Mar', 'S2_B2_Apr', 'S2_B2_May', 'S2_B2_Jun', 'S2_B2_Jul', 'S2_B2_Aug', 'S2_B2_Sep', 'S2_B2_Oct', 'S2_B2_Nov', 'S2_B2_Dec', 
             'S2_B3_Jan', 'S2_B3_Feb', 'S2_B3_Mar', 'S2_B3_Apr', 'S2_B3_May', 'S2_B3_Jun', 'S2_B3_Jul', 'S2_B3_Aug', 'S2_B3_Sep', 'S2_B3_Oct', 'S2_B3_Nov', 'S2_B3_Dec', 
             'S2_B4_Jan', 'S2_B4_Feb', 'S2_B4_Mar', 'S2_B4_Apr', 'S2_B4_May', 'S2_B4_Jun', 'S2_B4_Jul', 'S2_B4_Aug', 'S2_B4_Sep', 'S2_B4_Oct', 'S2_B4_Nov', 'S2_B4_Dec', 
             'S2_B5_Jan', 'S2_B5_Feb', 'S2_B5_Mar', 'S2_B5_Apr', 'S2_B5_May', 'S2_B5_Jun', 'S2_B5_Jul', 'S2_B5_Aug', 'S2_B5_Sep', 'S2_B5_Oct', 'S2_B5_Nov', 'S2_B5_Dec',
             'S2_B6_Jan', 'S2_B6_Feb', 'S2_B6_Mar', 'S2_B6_Apr', 'S2_B6_May', 'S2_B6_Jun', 'S2_B6_Jul', 'S2_B6_Aug', 'S2_B6_Sep', 'S2_B6_Oct', 'S2_B6_Nov', 'S2_B6_Dec',
             'S2_B7_Jan', 'S2_B7_Feb', 'S2_B7_Mar', 'S2_B7_Apr', 'S2_B7_May', 'S2_B7_Jun', 'S2_B7_Jul', 'S2_B7_Aug', 'S2_B7_Sep', 'S2_B7_Oct', 'S2_B7_Nov', 'S2_B7_Dec', 
             'S2_B8_Jan', 'S2_B8_Feb', 'S2_B8_Mar', 'S2_B8_Apr', 'S2_B8_May', 'S2_B8_Jun', 'S2_B8_Jul', 'S2_B8_Aug', 'S2_B8_Sep', 'S2_B8_Oct', 'S2_B8_Nov', 'S2_B8_Dec', 
             'S2_B8A_Jan', 'S2_B8A_Feb', 'S2_B8A_Mar', 'S2_B8A_Apr', 'S2_B8A_May', 'S2_B8A_Jun', 'S2_B8A_Jul', 'S2_B8A_Aug', 'S2_B8A_Sep', 'S2_B8A_Oct', 'S2_B8A_Nov', 'S2_B8A_Dec',  
             'S2_B11_Jan', 'S2_B11_Feb', 'S2_B11_Mar', 'S2_B11_Apr', 'S2_B11_May', 'S2_B11_Jun', 'S2_B11_Jul', 'S2_B11_Aug', 'S2_B11_Sep', 'S2_B11_Oct', 'S2_B11_Nov', 'S2_B11_Dec',
             'S2_B12_Jan', 'S2_B12_Feb', 'S2_B12_Mar', 'S2_B12_Apr', 'S2_B12_May', 'S2_B12_Jun', 'S2_B12_Jul', 'S2_B12_Aug', 'S2_B12_Sep', 'S2_B12_Oct', 'S2_B12_Nov', 'S2_B12_Dec', 
             'S2_CIgreen_Jan', 'S2_CIgreen_Feb', 'S2_CIgreen_Mar', 'S2_CIgreen_Apr', 'S2_CIgreen_May', 'S2_CIgreen_Jun', 'S2_CIgreen_Jul', 'S2_CIgreen_Aug', 'S2_CIgreen_Sep', 'S2_CIgreen_Oct', 'S2_CIgreen_Nov', 'S2_CIgreen_Dec', 
             'S2_CIre_Jan', 'S2_CIre_Feb', 'S2_CIre_Mar', 'S2_CIre_Apr', 'S2_CIre_May', 'S2_CIre_Jun', 'S2_CIre_Jul', 'S2_CIre_Aug', 'S2_CIre_Sep', 'S2_CIre_Oct', 'S2_CIre_Nov', 'S2_CIre_Dec', 
             'S2_DVI_Jan', 'S2_DVI_Feb', 'S2_DVI_Mar', 'S2_DVI_Apr', 'S2_DVI_May', 'S2_DVI_Jun', 'S2_DVI_Jul', 'S2_DVI_Aug', 'S2_DVI_Sep', 'S2_DVI_Oct', 'S2_DVI_Nov', 'S2_DVI_Dec', 
             'S2_EVI1_Jan', 'S2_EVI1_Feb', 'S2_EVI1_Mar', 'S2_EVI1_Apr', 'S2_EVI1_May', 'S2_EVI1_Jun', 'S2_EVI1_Jul', 'S2_EVI1_Aug', 'S2_EVI1_Sep', 'S2_EVI1_Oct', 'S2_EVI1_Nov', 'S2_EVI1_Dec', 
             'S2_EVI2_Jan', 'S2_EVI2_Feb', 'S2_EVI2_Mar', 'S2_EVI2_Apr', 'S2_EVI2_May', 'S2_EVI2_Jun', 'S2_EVI2_Jul', 'S2_EVI2_Aug', 'S2_EVI2_Sep', 'S2_EVI2_Oct', 'S2_EVI2_Nov', 'S2_EVI2_Dec', 
             'S2_EVIre1_Jan', 'S2_EVIre1_Feb', 'S2_EVIre1_Mar', 'S2_EVIre1_Apr', 'S2_EVIre1_May', 'S2_EVIre1_Jun', 'S2_EVIre1_Jul', 'S2_EVIre1_Aug', 'S2_EVIre1_Sep', 'S2_EVIre1_Oct', 'S2_EVIre1_Nov', 'S2_EVIre1_Dec', 
             'S2_EVIre2_Jan', 'S2_EVIre2_Feb', 'S2_EVIre2_Mar', 'S2_EVIre2_Apr', 'S2_EVIre2_May', 'S2_EVIre2_Jun', 'S2_EVIre2_Jul', 'S2_EVIre2_Aug', 'S2_EVIre2_Sep', 'S2_EVIre2_Oct', 'S2_EVIre2_Nov', 'S2_EVIre2_Dec', 
             'S2_EVIre3_Jan', 'S2_EVIre3_Feb', 'S2_EVIre3_Mar', 'S2_EVIre3_Apr', 'S2_EVIre3_May', 'S2_EVIre3_Jun', 'S2_EVIre3_Jul', 'S2_EVIre3_Aug', 'S2_EVIre3_Sep', 'S2_EVIre3_Oct', 'S2_EVIre3_Nov', 'S2_EVIre3_Dec', 
             'S2_GNDVI_Jan', 'S2_GNDVI_Feb', 'S2_GNDVI_Mar', 'S2_GNDVI_Apr', 'S2_GNDVI_May', 'S2_GNDVI_Jun', 'S2_GNDVI_Jul', 'S2_GNDVI_Aug', 'S2_GNDVI_Sep', 'S2_GNDVI_Oct', 'S2_GNDVI_Nov', 'S2_GNDVI_Dec', 
             'S2_IRECI_Jan', 'S2_IRECI_Feb', 'S2_IRECI_Mar', 'S2_IRECI_Apr', 'S2_IRECI_May', 'S2_IRECI_Jun', 'S2_IRECI_Jul', 'S2_IRECI_Aug', 'S2_IRECI_Sep', 'S2_IRECI_Oct', 'S2_IRECI_Nov', 'S2_IRECI_Dec',
             'S2_MCARI1_Jan', 'S2_MCARI1_Feb', 'S2_MCARI1_Mar', 'S2_MCARI1_Apr', 'S2_MCARI1_May', 'S2_MCARI1_Jun', 'S2_MCARI1_Jul', 'S2_MCARI1_Aug', 'S2_MCARI1_Sep', 'S2_MCARI1_Oct', 'S2_MCARI1_Nov', 'S2_MCARI1_Dec', 
             'S2_MCARI2_Jan', 'S2_MCARI2_Feb', 'S2_MCARI2_Mar', 'S2_MCARI2_Apr', 'S2_MCARI2_May', 'S2_MCARI2_Jun', 'S2_MCARI2_Jul', 'S2_MCARI2_Aug', 'S2_MCARI2_Sep', 'S2_MCARI2_Oct', 'S2_MCARI2_Nov', 'S2_MCARI2_Dec', 
             'S2_MCARI3_Jan', 'S2_MCARI3_Feb', 'S2_MCARI3_Mar', 'S2_MCARI3_Apr', 'S2_MCARI3_May', 'S2_MCARI3_Jun', 'S2_MCARI3_Jul', 'S2_MCARI3_Aug', 'S2_MCARI3_Sep', 'S2_MCARI3_Oct', 'S2_MCARI3_Nov', 'S2_MCARI3_Dec', 
             'S2_MTCI1_Jan', 'S2_MTCI1_Feb', 'S2_MTCI1_Mar', 'S2_MTCI1_Apr', 'S2_MTCI1_May', 'S2_MTCI1_Jun', 'S2_MTCI1_Jul', 'S2_MTCI1_Aug', 'S2_MTCI1_Sep', 'S2_MTCI1_Oct', 'S2_MTCI1_Nov', 'S2_MTCI1_Dec', 
             'S2_MTCI2_Jan', 'S2_MTCI2_Feb', 'S2_MTCI2_Mar', 'S2_MTCI2_Apr', 'S2_MTCI2_May', 'S2_MTCI2_Jun', 'S2_MTCI2_Jul', 'S2_MTCI2_Aug', 'S2_MTCI2_Sep', 'S2_MTCI2_Oct', 'S2_MTCI2_Nov', 'S2_MTCI2_Dec', 
             'S2_MTCI3_Jan', 'S2_MTCI3_Feb', 'S2_MTCI3_Mar', 'S2_MTCI3_Apr', 'S2_MTCI3_May', 'S2_MTCI3_Jun', 'S2_MTCI3_Jul', 'S2_MTCI3_Aug', 'S2_MTCI3_Sep', 'S2_MTCI3_Oct', 'S2_MTCI3_Nov', 'S2_MTCI3_Dec', 
             'S2_NDI45_Jan', 'S2_NDI45_Feb', 'S2_NDI45_Mar', 'S2_NDI45_Apr', 'S2_NDI45_May', 'S2_NDI45_Jun', 'S2_NDI45_Jul', 'S2_NDI45_Aug', 'S2_NDI45_Sep', 'S2_NDI45_Oct', 'S2_NDI45_Nov', 'S2_NDI45_Dec', 
             'S2_NDRE1_Jan', 'S2_NDRE1_Feb', 'S2_NDRE1_Mar', 'S2_NDRE1_Apr', 'S2_NDRE1_May', 'S2_NDRE1_Jun', 'S2_NDRE1_Jul', 'S2_NDRE1_Aug', 'S2_NDRE1_Sep', 'S2_NDRE1_Oct', 'S2_NDRE1_Nov', 'S2_NDRE1_Dec',
             'S2_NDRE2_Jan', 'S2_NDRE2_Feb', 'S2_NDRE2_Mar', 'S2_NDRE2_Apr', 'S2_NDRE2_May', 'S2_NDRE2_Jun', 'S2_NDRE2_Jul', 'S2_NDRE2_Aug', 'S2_NDRE2_Sep', 'S2_NDRE2_Oct', 'S2_NDRE2_Nov', 'S2_NDRE2_Dec', 
             'S2_NDRE3_Jan', 'S2_NDRE3_Feb', 'S2_NDRE3_Mar', 'S2_NDRE3_Apr', 'S2_NDRE3_May', 'S2_NDRE3_Jun', 'S2_NDRE3_Jul', 'S2_NDRE3_Aug', 'S2_NDRE3_Sep', 'S2_NDRE3_Oct', 'S2_NDRE3_Nov', 'S2_NDRE3_Dec',
             'S2_NDVI56_Jan', 'S2_NDVI56_Feb', 'S2_NDVI56_Mar', 'S2_NDVI56_Apr', 'S2_NDVI56_May', 'S2_NDVI56_Jun', 'S2_NDVI56_Jul', 'S2_NDVI56_Aug', 'S2_NDVI56_Sep', 'S2_NDVI56_Oct', 'S2_NDVI56_Nov', 'S2_NDVI56_Dec', 
             'S2_NDVI57_Jan', 'S2_NDVI57_Feb', 'S2_NDVI57_Mar', 'S2_NDVI57_Apr', 'S2_NDVI57_May', 'S2_NDVI57_Jun', 'S2_NDVI57_Jul', 'S2_NDVI57_Aug', 'S2_NDVI57_Sep', 'S2_NDVI57_Oct', 'S2_NDVI57_Nov', 'S2_NDVI57_Dec', 
             'S2_NDVI68a_Jan', 'S2_NDVI68a_Feb', 'S2_NDVI68a_Mar', 'S2_NDVI68a_Apr', 'S2_NDVI68a_May', 'S2_NDVI68a_Jun', 'S2_NDVI68a_Jul', 'S2_NDVI68a_Aug', 'S2_NDVI68a_Sep', 'S2_NDVI68a_Oct', 'S2_NDVI68a_Nov', 'S2_NDVI68a_Dec',
             'S2_NDVI78a_Jan', 'S2_NDVI78a_Feb', 'S2_NDVI78a_Mar', 'S2_NDVI78a_Apr', 'S2_NDVI78a_May', 'S2_NDVI78a_Jun', 'S2_NDVI78a_Jul', 'S2_NDVI78a_Aug', 'S2_NDVI78a_Sep', 'S2_NDVI78a_Oct', 'S2_NDVI78a_Nov', 'S2_NDVI78a_Dec', 
             'S2_NDWI1_Jan', 'S2_NDWI1_Feb', 'S2_NDWI1_Mar', 'S2_NDWI1_Apr', 'S2_NDWI1_May', 'S2_NDWI1_Jun', 'S2_NDWI1_Jul', 'S2_NDWI1_Aug', 'S2_NDWI1_Sep', 'S2_NDWI1_Oct', 'S2_NDWI1_Nov', 'S2_NDWI1_Dec', 
             'S2_NDWI2_Jan', 'S2_NDWI2_Feb', 'S2_NDWI2_Mar', 'S2_NDWI2_Apr', 'S2_NDWI2_May', 'S2_NDWI2_Jun', 'S2_NDWI2_Jul', 'S2_NDWI2_Aug', 'S2_NDWI2_Sep', 'S2_NDWI2_Oct', 'S2_NDWI2_Nov', 'S2_NDWI2_Dec',
             'S2_NIRv_Jan', 'S2_NIRv_Feb', 'S2_NIRv_Mar', 'S2_NIRv_Apr', 'S2_NIRv_May', 'S2_NIRv_Jun', 'S2_NIRv_Jul', 'S2_NIRv_Aug', 'S2_NIRv_Sep', 'S2_NIRv_Oct', 'S2_NIRv_Nov', 'S2_NIRv_Dec',
             'S2_NLI_Jan', 'S2_NLI_Feb', 'S2_NLI_Mar', 'S2_NLI_Apr', 'S2_NLI_May', 'S2_NLI_Jun', 'S2_NLI_Jul', 'S2_NLI_Aug', 'S2_NLI_Sep', 'S2_NLI_Oct', 'S2_NLI_Nov', 'S2_NLI_Dec', 
             'S2_OSAVI_Jan', 'S2_OSAVI_Feb', 'S2_OSAVI_Mar', 'S2_OSAVI_Apr', 'S2_OSAVI_May', 'S2_OSAVI_Jun', 'S2_OSAVI_Jul', 'S2_OSAVI_Aug', 'S2_OSAVI_Sep', 'S2_OSAVI_Oct', 'S2_OSAVI_Nov', 'S2_OSAVI_Dec', 
             'S2_PSSRa_Jan', 'S2_PSSRa_Feb', 'S2_PSSRa_Mar', 'S2_PSSRa_Apr', 'S2_PSSRa_May', 'S2_PSSRa_Jun', 'S2_PSSRa_Jul', 'S2_PSSRa_Aug', 'S2_PSSRa_Sep', 'S2_PSSRa_Oct', 'S2_PSSRa_Nov', 'S2_PSSRa_Dec',
             'S2_SAVI_Jan', 'S2_SAVI_Feb', 'S2_SAVI_Mar', 'S2_SAVI_Apr', 'S2_SAVI_May', 'S2_SAVI_Jun', 'S2_SAVI_Jul', 'S2_SAVI_Aug', 'S2_SAVI_Sep', 'S2_SAVI_Oct', 'S2_SAVI_Nov', 'S2_SAVI_Dec',
             'S2_SR_Jan', 'S2_SR_Feb', 'S2_SR_Mar', 'S2_SR_Apr', 'S2_SR_May', 'S2_SR_Jun', 'S2_SR_Jul', 'S2_SR_Aug', 'S2_SR_Sep', 'S2_SR_Oct', 'S2_SR_Nov', 'S2_SR_Dec',
             'S2_kNDVI_Jan', 'S2_kNDVI_Feb', 'S2_kNDVI_Mar', 'S2_kNDVI_Apr', 'S2_kNDVI_May', 'S2_kNDVI_Jun', 'S2_kNDVI_Jul', 'S2_kNDVI_Aug', 'S2_kNDVI_Sep', 'S2_kNDVI_Oct', 'S2_kNDVI_Nov', 'S2_kNDVI_Dec',
             
             'P2_HH+HV_Jan', 'P2_HH+HV_Feb', 'P2_HH+HV_Mar', 'P2_HH+HV_Apr', 'P2_HH+HV_May', 'P2_HH+HV_Jun', 'P2_HH+HV_Jul', 'P2_HH+HV_Aug', 'P2_HH+HV_Sep', 'P2_HH+HV_Oct', 'P2_HH+HV_Nov', 'P2_HH+HV_Dec',
             'P2_HH-HV_Jan', 'P2_HH-HV_Feb', 'P2_HH-HV_Mar', 'P2_HH-HV_Apr', 'P2_HH-HV_May', 'P2_HH-HV_Jun', 'P2_HH-HV_Jul', 'P2_HH-HV_Aug', 'P2_HH-HV_Sep', 'P2_HH-HV_Oct', 'P2_HH-HV_Nov', 'P2_HH-HV_Dec',
             'P2_HH_Jan', 'P2_HH_Feb', 'P2_HH_Mar', 'P2_HH_Apr', 'P2_HH_May', 'P2_HH_Jun', 'P2_HH_Jul', 'P2_HH_Aug', 'P2_HH_Sep', 'P2_HH_Oct', 'P2_HH_Nov', 'P2_HH_Dec', 
             'P2_HH_asm_Jan', 'P2_HH_asm_Feb', 'P2_HH_asm_Mar', 'P2_HH_asm_Apr', 'P2_HH_asm_May', 'P2_HH_asm_Jun', 'P2_HH_asm_Jul', 'P2_HH_asm_Aug', 'P2_HH_asm_Sep', 'P2_HH_asm_Oct', 'P2_HH_asm_Nov', 'P2_HH_asm_Dec', 
             'P2_HH_con_Jan', 'P2_HH_con_Feb', 'P2_HH_con_Mar', 'P2_HH_con_Apr', 'P2_HH_con_May', 'P2_HH_con_Jun', 'P2_HH_con_Jul', 'P2_HH_con_Aug', 'P2_HH_con_Sep', 'P2_HH_con_Oct', 'P2_HH_con_Nov', 'P2_HH_con_Dec',
             'P2_HH_corr_Jan', 'P2_HH_corr_Feb', 'P2_HH_corr_Mar', 'P2_HH_corr_Apr', 'P2_HH_corr_May', 'P2_HH_corr_Jun', 'P2_HH_corr_Jul', 'P2_HH_corr_Aug', 'P2_HH_corr_Sep', 'P2_HH_corr_Oct', 'P2_HH_corr_Nov', 'P2_HH_corr_Dec',
             'P2_HH_dent_Jan', 'P2_HH_dent_Feb', 'P2_HH_dent_Mar', 'P2_HH_dent_Apr', 'P2_HH_dent_May', 'P2_HH_dent_Jun', 'P2_HH_dent_Jul', 'P2_HH_dent_Aug', 'P2_HH_dent_Sep', 'P2_HH_dent_Oct', 'P2_HH_dent_Nov', 'P2_HH_dent_Dec', 
             'P2_HH_diss_Jan', 'P2_HH_diss_Feb', 'P2_HH_diss_Mar', 'P2_HH_diss_Apr', 'P2_HH_diss_May', 'P2_HH_diss_Jun', 'P2_HH_diss_Jul', 'P2_HH_diss_Aug', 'P2_HH_diss_Sep', 'P2_HH_diss_Oct', 'P2_HH_diss_Nov', 'P2_HH_diss_Dec',
             'P2_HH_dvar_Jan', 'P2_HH_dvar_Feb', 'P2_HH_dvar_Mar', 'P2_HH_dvar_Apr', 'P2_HH_dvar_May', 'P2_HH_dvar_Jun', 'P2_HH_dvar_Jul', 'P2_HH_dvar_Aug', 'P2_HH_dvar_Sep', 'P2_HH_dvar_Oct', 'P2_HH_dvar_Nov', 'P2_HH_dvar_Dec', 
             'P2_HH_ent_Jan', 'P2_HH_ent_Feb', 'P2_HH_ent_Mar', 'P2_HH_ent_Apr', 'P2_HH_ent_May', 'P2_HH_ent_Jun', 'P2_HH_ent_Jul', 'P2_HH_ent_Aug', 'P2_HH_ent_Sep', 'P2_HH_ent_Oct', 'P2_HH_ent_Nov', 'P2_HH_ent_Dec',
             'P2_HH_homo_Jan', 'P2_HH_homo_Feb', 'P2_HH_homo_Mar', 'P2_HH_homo_Apr', 'P2_HH_homo_May', 'P2_HH_homo_Jun', 'P2_HH_homo_Jul', 'P2_HH_homo_Aug', 'P2_HH_homo_Sep', 'P2_HH_homo_Oct', 'P2_HH_homo_Nov', 'P2_HH_homo_Dec',
             'P2_HH_imcorr1_Jan', 'P2_HH_imcorr1_Feb', 'P2_HH_imcorr1_Mar', 'P2_HH_imcorr1_Apr', 'P2_HH_imcorr1_May', 'P2_HH_imcorr1_Jun', 'P2_HH_imcorr1_Jul', 'P2_HH_imcorr1_Aug', 'P2_HH_imcorr1_Sep', 'P2_HH_imcorr1_Oct', 'P2_HH_imcorr1_Nov', 'P2_HH_imcorr1_Dec',
             'P2_HH_imcorr2_Jan', 'P2_HH_imcorr2_Feb', 'P2_HH_imcorr2_Mar', 'P2_HH_imcorr2_Apr', 'P2_HH_imcorr2_May', 'P2_HH_imcorr2_Jun', 'P2_HH_imcorr2_Jul', 'P2_HH_imcorr2_Aug', 'P2_HH_imcorr2_Sep', 'P2_HH_imcorr2_Oct', 'P2_HH_imcorr2_Nov', 'P2_HH_imcorr2_Dec', 
             'P2_HH_inertia_Jan', 'P2_HH_inertia_Feb', 'P2_HH_inertia_Mar', 'P2_HH_inertia_Apr', 'P2_HH_inertia_May', 'P2_HH_inertia_Jun', 'P2_HH_inertia_Jul', 'P2_HH_inertia_Aug', 'P2_HH_inertia_Sep', 'P2_HH_inertia_Oct', 'P2_HH_inertia_Nov', 'P2_HH_inertia_Dec', 
             'P2_HH_prom_Jan', 'P2_HH_prom_Feb', 'P2_HH_prom_Mar', 'P2_HH_prom_Apr', 'P2_HH_prom_May', 'P2_HH_prom_Jun', 'P2_HH_prom_Jul', 'P2_HH_prom_Aug', 'P2_HH_prom_Sep', 'P2_HH_prom_Oct', 'P2_HH_prom_Nov', 'P2_HH_prom_Dec', 
             'P2_HH_savg_Jan', 'P2_HH_savg_Feb', 'P2_HH_savg_Mar', 'P2_HH_savg_Apr', 'P2_HH_savg_May', 'P2_HH_savg_Jun', 'P2_HH_savg_Jul', 'P2_HH_savg_Aug', 'P2_HH_savg_Sep', 'P2_HH_savg_Oct', 'P2_HH_savg_Nov', 'P2_HH_savg_Dec',
             'P2_HH_sent_Jan', 'P2_HH_sent_Feb', 'P2_HH_sent_Mar', 'P2_HH_sent_Apr', 'P2_HH_sent_May', 'P2_HH_sent_Jun', 'P2_HH_sent_Jul', 'P2_HH_sent_Aug', 'P2_HH_sent_Sep', 'P2_HH_sent_Oct', 'P2_HH_sent_Nov', 'P2_HH_sent_Dec',
             'P2_HH_shade_Jan', 'P2_HH_shade_Feb', 'P2_HH_shade_Mar', 'P2_HH_shade_Apr', 'P2_HH_shade_May', 'P2_HH_shade_Jun', 'P2_HH_shade_Jul', 'P2_HH_shade_Aug', 'P2_HH_shade_Sep', 'P2_HH_shade_Oct', 'P2_HH_shade_Nov', 'P2_HH_shade_Dec',
             'P2_HH_svar_Jan', 'P2_HH_svar_Feb', 'P2_HH_svar_Mar', 'P2_HH_svar_Apr', 'P2_HH_svar_May', 'P2_HH_svar_Jun', 'P2_HH_svar_Jul', 'P2_HH_svar_Aug', 'P2_HH_svar_Sep', 'P2_HH_svar_Oct', 'P2_HH_svar_Nov', 'P2_HH_svar_Dec', 
             'P2_HH_var_Jan', 'P2_HH_var_Feb', 'P2_HH_var_Mar', 'P2_HH_var_Apr', 'P2_HH_var_May', 'P2_HH_var_Jun', 'P2_HH_var_Jul', 'P2_HH_var_Aug', 'P2_HH_var_Sep', 'P2_HH_var_Oct', 'P2_HH_var_Nov', 'P2_HH_var_Dec', 
             'P2_HV/HH_Jan', 'P2_HV/HH_Feb', 'P2_HV/HH_Mar', 'P2_HV/HH_Apr', 'P2_HV/HH_May', 'P2_HV/HH_Jun', 'P2_HV/HH_Jul', 'P2_HV/HH_Aug', 'P2_HV/HH_Sep', 'P2_HV/HH_Oct', 'P2_HV/HH_Nov', 'P2_HV/HH_Dec', 
             'P2_HV_Jan', 'P2_HV_Feb', 'P2_HV_Mar', 'P2_HV_Apr', 'P2_HV_May', 'P2_HV_Jun', 'P2_HV_Jul', 'P2_HV_Aug', 'P2_HV_Sep', 'P2_HV_Oct', 'P2_HV_Nov', 'P2_HV_Dec', 
             'P2_HV_asm_Jan', 'P2_HV_asm_Feb', 'P2_HV_asm_Mar', 'P2_HV_asm_Apr', 'P2_HV_asm_May', 'P2_HV_asm_Jun', 'P2_HV_asm_Jul', 'P2_HV_asm_Aug', 'P2_HV_asm_Sep', 'P2_HV_asm_Oct', 'P2_HV_asm_Nov', 'P2_HV_asm_Dec', 
             'P2_HV_con_Jan', 'P2_HV_con_Feb', 'P2_HV_con_Mar', 'P2_HV_con_Apr', 'P2_HV_con_May', 'P2_HV_con_Jun', 'P2_HV_con_Jul', 'P2_HV_con_Aug', 'P2_HV_con_Sep', 'P2_HV_con_Oct', 'P2_HV_con_Nov', 'P2_HV_con_Dec', 
             'P2_HV_corr_Jan', 'P2_HV_corr_Feb', 'P2_HV_corr_Mar', 'P2_HV_corr_Apr', 'P2_HV_corr_May', 'P2_HV_corr_Jun', 'P2_HV_corr_Jul', 'P2_HV_corr_Aug', 'P2_HV_corr_Sep', 'P2_HV_corr_Oct', 'P2_HV_corr_Nov', 'P2_HV_corr_Dec', 
             'P2_HV_dent_Jan', 'P2_HV_dent_Feb', 'P2_HV_dent_Mar', 'P2_HV_dent_Apr', 'P2_HV_dent_May', 'P2_HV_dent_Jun', 'P2_HV_dent_Jul', 'P2_HV_dent_Aug', 'P2_HV_dent_Sep', 'P2_HV_dent_Oct', 'P2_HV_dent_Nov', 'P2_HV_dent_Dec', 
             'P2_HV_diss_Jan', 'P2_HV_diss_Feb', 'P2_HV_diss_Mar', 'P2_HV_diss_Apr', 'P2_HV_diss_May', 'P2_HV_diss_Jun', 'P2_HV_diss_Jul', 'P2_HV_diss_Aug', 'P2_HV_diss_Sep', 'P2_HV_diss_Oct', 'P2_HV_diss_Nov', 'P2_HV_diss_Dec',
             'P2_HV_dvar_Jan', 'P2_HV_dvar_Feb', 'P2_HV_dvar_Mar', 'P2_HV_dvar_Apr', 'P2_HV_dvar_May', 'P2_HV_dvar_Jun', 'P2_HV_dvar_Jul', 'P2_HV_dvar_Aug', 'P2_HV_dvar_Sep', 'P2_HV_dvar_Oct', 'P2_HV_dvar_Nov', 'P2_HV_dvar_Dec',
             'P2_HV_ent_Jan', 'P2_HV_ent_Feb', 'P2_HV_ent_Mar', 'P2_HV_ent_Apr', 'P2_HV_ent_May', 'P2_HV_ent_Jun', 'P2_HV_ent_Jul', 'P2_HV_ent_Aug', 'P2_HV_ent_Sep', 'P2_HV_ent_Oct', 'P2_HV_ent_Nov', 'P2_HV_ent_Dec', 
             'P2_HV_homo_Jan', 'P2_HV_homo_Feb', 'P2_HV_homo_Mar', 'P2_HV_homo_Apr', 'P2_HV_homo_May', 'P2_HV_homo_Jun', 'P2_HV_homo_Jul', 'P2_HV_homo_Aug', 'P2_HV_homo_Sep', 'P2_HV_homo_Oct', 'P2_HV_homo_Nov', 'P2_HV_homo_Dec',
             'P2_HV_imcorr1_Jan', 'P2_HV_imcorr1_Feb', 'P2_HV_imcorr1_Mar', 'P2_HV_imcorr1_Apr', 'P2_HV_imcorr1_May', 'P2_HV_imcorr1_Jun', 'P2_HV_imcorr1_Jul', 'P2_HV_imcorr1_Aug', 'P2_HV_imcorr1_Sep', 'P2_HV_imcorr1_Oct', 'P2_HV_imcorr1_Nov', 'P2_HV_imcorr1_Dec', 
             'P2_HV_imcorr2_Jan', 'P2_HV_imcorr2_Feb', 'P2_HV_imcorr2_Mar', 'P2_HV_imcorr2_Apr', 'P2_HV_imcorr2_May', 'P2_HV_imcorr2_Jun', 'P2_HV_imcorr2_Jul', 'P2_HV_imcorr2_Aug', 'P2_HV_imcorr2_Sep', 'P2_HV_imcorr2_Oct', 'P2_HV_imcorr2_Nov', 'P2_HV_imcorr2_Dec', 
             'P2_HV_inertia_Jan', 'P2_HV_inertia_Feb', 'P2_HV_inertia_Mar', 'P2_HV_inertia_Apr', 'P2_HV_inertia_May', 'P2_HV_inertia_Jun', 'P2_HV_inertia_Jul', 'P2_HV_inertia_Aug', 'P2_HV_inertia_Sep', 'P2_HV_inertia_Oct', 'P2_HV_inertia_Nov', 'P2_HV_inertia_Dec',
             'P2_HV_prom_Jan', 'P2_HV_prom_Feb', 'P2_HV_prom_Mar', 'P2_HV_prom_Apr', 'P2_HV_prom_May', 'P2_HV_prom_Jun', 'P2_HV_prom_Jul', 'P2_HV_prom_Aug', 'P2_HV_prom_Sep', 'P2_HV_prom_Oct', 'P2_HV_prom_Nov', 'P2_HV_prom_Dec', 
             'P2_HV_savg_Jan', 'P2_HV_savg_Feb', 'P2_HV_savg_Mar', 'P2_HV_savg_Apr', 'P2_HV_savg_May', 'P2_HV_savg_Jun', 'P2_HV_savg_Jul', 'P2_HV_savg_Aug', 'P2_HV_savg_Sep', 'P2_HV_savg_Oct', 'P2_HV_savg_Nov', 'P2_HV_savg_Dec', 
             'P2_HV_sent_Jan', 'P2_HV_sent_Feb', 'P2_HV_sent_Mar', 'P2_HV_sent_Apr', 'P2_HV_sent_May', 'P2_HV_sent_Jun', 'P2_HV_sent_Jul', 'P2_HV_sent_Aug', 'P2_HV_sent_Sep', 'P2_HV_sent_Oct', 'P2_HV_sent_Nov', 'P2_HV_sent_Dec', 
             'P2_HV_shade_Jan', 'P2_HV_shade_Feb', 'P2_HV_shade_Mar', 'P2_HV_shade_Apr', 'P2_HV_shade_May', 'P2_HV_shade_Jun', 'P2_HV_shade_Jul', 'P2_HV_shade_Aug', 'P2_HV_shade_Sep', 'P2_HV_shade_Oct', 'P2_HV_shade_Nov', 'P2_HV_shade_Dec', 
             'P2_HV_svar_Jan', 'P2_HV_svar_Feb', 'P2_HV_svar_Mar', 'P2_HV_svar_Apr', 'P2_HV_svar_May', 'P2_HV_svar_Jun', 'P2_HV_svar_Jul', 'P2_HV_svar_Aug', 'P2_HV_svar_Sep', 'P2_HV_svar_Oct', 'P2_HV_svar_Nov', 'P2_HV_svar_Dec',
             'P2_HV_var_Jan', 'P2_HV_var_Feb', 'P2_HV_var_Mar', 'P2_HV_var_Apr', 'P2_HV_var_May', 'P2_HV_var_Jun', 'P2_HV_var_Jul', 'P2_HV_var_Aug', 'P2_HV_var_Sep', 'P2_HV_var_Oct', 'P2_HV_var_Nov', 'P2_HV_var_Dec'
           ]

    # load data: train, val and test
    for dataset in['training', 'validation', 'test']:
        if dataset == 'training':
            train_path = "../data/train_month_median_pivot.csv"  
            
            X_train, y_train = prepare_df(train_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_train, dataset, directory)
            X_train_scaled, y_train_scaled,  feature_scaler, target_scaler = standard_df(X_train, y_train, directory)
            # display(X_train_scaled[:1])
            # display(y_train_scaled[:1])
            # split cover
            X_train_Croplands, y_train_Croplands = prepare_df(train_path, cols, dataset, directory, cover_type="Croplands")
            X_train_Croplands_scaled = feature_scaler.transform(X_train_Croplands)
            y_train_Croplands_scaled = target_scaler.transform(y_train_Croplands.values.reshape(-1, 1))
            
            X_train_Forests, y_train_Forests = prepare_df(train_path, cols, dataset, directory, cover_type="Forests")
            X_train_Forests_scaled = feature_scaler.transform(X_train_Forests)
            y_train_Forests_scaled = target_scaler.transform(y_train_Forests.values.reshape(-1, 1))
            
            X_train_Savannas, y_train_Savannas= prepare_df(train_path, cols, dataset, directory, cover_type="Savannas")
            X_train_Savannas_scaled = feature_scaler.transform(X_train_Savannas)
            y_train_Savannas_scaled = target_scaler.transform(y_train_Savannas.values.reshape(-1, 1))
            
            X_train_Shrub_grass_lands, y_train_Shrub_grass_lands = prepare_df(train_path, cols, dataset, directory, cover_type="Shrub_grass_lands")
            X_train_Shrub_grass_lands_scaled = feature_scaler.transform(X_train_Shrub_grass_lands)
            y_train_Shrub_grass_lands_scaled = target_scaler.transform(y_train_Shrub_grass_lands.values.reshape(-1, 1))
            
       
        elif dataset == "validation":
            val_path ="../data/val_month_median_pivot.csv" 
            X_val, y_val = prepare_df(val_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_val, dataset, directory)
            X_val_scaled = feature_scaler.transform(X_val)
            y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
            
        else:
            test_path = "../data/test_month_median_pivot.csv" 
            
            X_test, y_test = prepare_df(test_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_test, dataset, directory)
            X_test_scaled = feature_scaler.transform(X_test)
            y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
            # split cover
            X_test_Croplands, y_test_Croplands = prepare_df(test_path, cols, dataset, directory, cover_type="Croplands")
            X_test_Croplands_scaled = feature_scaler.transform(X_test_Croplands)
            y_test_Croplands_scaled = target_scaler.transform(y_test_Croplands.values.reshape(-1, 1))
            
            X_test_Forests, y_test_Forests = prepare_df(test_path, cols, dataset, directory, cover_type="Forests")
            X_test_Forests_scaled = feature_scaler.transform(X_test_Forests)
            y_test_Forests_scaled = target_scaler.transform(y_test_Forests.values.reshape(-1, 1))
            
            X_test_Savannas, y_test_Savannas= prepare_df(test_path, cols, dataset, directory, cover_type="Savannas")
            X_test_Savannas_scaled = feature_scaler.transform(X_test_Savannas)
            y_test_Savannas_scaled = target_scaler.transform(y_test_Savannas.values.reshape(-1, 1))
            
            X_test_Shrub_grass_lands, y_test_Shrub_grass_lands = prepare_df(test_path, cols, dataset, directory, cover_type="Shrub_grass_lands")
            X_test_Shrub_grass_lands_scaled = feature_scaler.transform(X_test_Shrub_grass_lands)
            y_test_Shrub_grass_lands_scaled = target_scaler.transform(y_test_Shrub_grass_lands.values.reshape(-1, 1))    
            
    print(f"\n⚡X_train.head(1): \n{X_train.head(1)}")

    
    # --- Configuration (Non-tunable) ---
    INPUT_SIZE = len(cols) # Number of input features
    OUTPUT_SIZE = 1 # Number of output neurons 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔋 Model was trained via {DEVICE}")
    
    # train_scaled to tensor
    train_tensor = prepare_tensor(X_train_scaled, y_train_scaled)
    X_train_tensor, y_train_tensor = train_tensor.tensors  # tensors is a tuple (X, y)
    print("\n⚡X_train_tensor shape:", X_train_tensor.shape)
    print(f'X_train_tensor[:1]: \n{X_train_tensor[:1]}')
    
    train_tensor_Croplands = prepare_tensor(X_train_Croplands_scaled, y_train_Croplands_scaled)
    train_tensor_Forests = prepare_tensor(X_train_Forests_scaled, y_train_Forests_scaled)
    train_tensor_Savannas = prepare_tensor(X_train_Savannas_scaled, y_train_Savannas_scaled)
    train_tensor_Shrub_grass_lands = prepare_tensor(X_train_Shrub_grass_lands_scaled, y_train_Shrub_grass_lands_scaled)
    
    # val_scaled to tensor        
    val_tensor = prepare_tensor(X_val_scaled, y_val_scaled)
    
    # test_scaled to tensor  
    test_tensor = prepare_tensor(X_test_scaled, y_test_scaled)
    test_tensor_Croplands = prepare_tensor(X_test_Croplands_scaled, y_test_Croplands_scaled)
    test_tensor_Forests = prepare_tensor(X_test_Forests_scaled, y_test_Forests_scaled)
    test_tensor_Savannas = prepare_tensor(X_test_Savannas_scaled, y_test_Savannas_scaled)
    test_tensor_Shrub_grass_lands = prepare_tensor(X_test_Shrub_grass_lands_scaled, y_test_Shrub_grass_lands_scaled)
    # training starts
    starttime = datetime.now()
    
    # --- 5. Run Optimization ---
    # Set up the Optuna study using the TPESampler and 
    study = optuna.create_study(
            # sampler=optuna.samplers.RandomSampler(seed=42),
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=20),
            sampler=optuna.samplers.TPESampler(n_startup_trials=100),
            # sampler=optuna.samplers.GPSampler(n_startup_trials=100),
            # pruner=optuna.pruners.NopPruner(), # Every trial will train until early stopping or EPOCHS.
            direction='minimize'
        )
    # optimization
    from functools import partial
    study.optimize(partial(objective, save_dir=directory), 
                   n_trials=20,
                   n_jobs=-1
                  )  
    #end_time
    endtime = datetime.now()
    used_time = endtime - starttime
    
    print(f"\n--xgb training time--: {used_time }")
    
    print("\nOptimization Complete.")
    print(f"Best Value (MSE): {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)
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
    compute_and_plot_shap(final_model, X_train_tensor, cols, directory)
    print("\nExecution done!")      
