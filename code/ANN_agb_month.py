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
from torchsummary import summary

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
def prepare_df(csv_path, col, datatype, save_dir):
    """
    Load data from a CSV file, preprocess it, and convert it into a df for rf.

    Parameters:
    - csv_path: Path to the CSV file.
    - col: List of column names for features.
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X: DataFrame of features.
    - y: Series of target variable.
    """
    
    # Load the data
    df = pd.read_csv(csv_path)


    # Separate features and target variable for training and testing
    X = df[cols]
    y = df["AGBD"]
    
    print(f"\nTotal {datatype} samples=:", len(X))

    print("\nDataframe feature and target split successfully!")
    with open(f"{save_dir}_used_features.txt", "w") as file:
         file.write(f"used features: \n{col}")
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
    Load data from a CSV file, preprocess it, and convert it into a df for rf.

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


# Updated prepare_tensor (see detailed version above)
def prepare_tensor(X_scaled: np.ndarray, y_scaled: np.ndarray, seq_len: int = 64):
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(-1)  # (n_samples, 64, 1)
    y_tensor = torch.FloatTensor(y_scaled)                # (n_samples, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"✅ GRU-compatible TensorDataset created → {len(dataset)} samples, "
          f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
    
    return dataset

    
# Model Definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # ← critical fix
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
def objective(trial, save_dir):
    # --- 1. Define Search Space ---
    hidden_size =  trial.suggest_categorical(f"hidden_size", [16, 32, 64, 128, 256, 512,1024])
    num_layers = trial.suggest_int("num_layers", 1, 8, step=1)
    initial_lr = trial.suggest_categorical("lr", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    dropout = trial.suggest_float("dropout",  0.0, 0.6, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512,1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0])  # log scale

    # --- 2. Setup Trial Objects ---
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader =  DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    model = GRUModel(INPUT_SIZE, 
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
    # Option 1: Simple string representation (already done above)
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    print(model)
    print("\nDetailed layer summary:")
    print("-"*50)
    # Option 2: More detailed parameter info using torchsummary (recommended)
    try:
        from torchsummary import summary
        summary(model, input_size=(64, INPUT_SIZE)) # (batch_size is optional, here we omit it)
    except ImportError:
        print("Note: Install torchsummary (`pip install torchsummary`) for detailed summary.")
    
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
        min_lr=1e-6,
    )
    
    # --- 5. Early Stopping Settings ---
    EPOCHS = 200
    patience = 20
    min_delta = 1e-5
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
        if avg_val_mse < best_val_loss - min_delta:
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
    model = GRUModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers = num_layers,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(DEVICE)
    
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("\n" + "="*50)
        print("MODEL ARCHITECTURE (PyTorch String)")
        print("="*50)
        print(model)
        
        print("\n" + "="*50)
        print("DETAILED LAYER SUMMARY (torchsummary)")
        print("="*50)
        try:
            from torchsummary import summary
            summary(
                model,
                input_size=(64, INPUT_SIZE),
                device=str(DEVICE).split(':')[0]
            )
        except Exception as e:
            print(f"Could not generate torchsummary: {e}")
    
    # Write everything to file
    output_path = f"{save_dir}_model_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())
    print(f"Model summary saved to {output_path}")
    print("\n--- Model Summary Visual ---")
    summary(model,  input_size=(64, INPUT_SIZE),  device=str(DEVICE).split(':')[0])

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
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    # -----------------------------
    # Early stopping
    # -----------------------------
    EPOCHS = 200
    patience = 20
    min_delta = 1e-5
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
        if avg_val_mse < best_val_loss - min_delta:
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


# SHAP based feature importance 
def compute_and_plot_shap(final_model, X_train_tensor, cols, save_dir):
    """
    Compute and plot SHAP values with automatic GPU/CPU selection.
    
    Parameters:
    final_model: Trained model for SHAP explanation
    X_train_tensor: X_train_tensor
    save_dir (str): Directory to save the output files
    """

    old_cudnn = torch.backends.cudnn.enabled
    old_mode = final_model.training
    
    # Setup for deterministic explanation (no dropout, no cuDNN restriction)
    torch.backends.cudnn.enabled = False
    final_model.eval()

    # random select certain samples in X_train_tensor as background
    idx = torch.randperm(X_train_tensor.size(0))[:100]
    background = X_train_tensor[idx].to(DEVICE)

    # Explainer and computation
    explainer = shap.GradientExplainer(final_model, background)
    shap_values_array = explainer.shap_values(X_train_tensor.to(DEVICE))
    
    print(f"🚀shap_values_array.shape: {shap_values_array.shape}")
    # print(shap_values_array)  # Still noisy; better to summarize/visualize
    
    # Restore original states
    torch.backends.cudnn.enabled = old_cudnn
    if old_mode:
        final_model.train()
    else:
        final_model.eval()
  
    shap_2d = np.squeeze(shap_values_array)                  # Remove all singleton dimensions
    if len(shap_2d.shape) > 2:
        # Flatten everything except the sample dimension (handles extra output/feature singletons)
        shap_2d = np.reshape(shap_2d, (shap_2d.shape[0], -1))
    
    print(f"🚀shap_2d.shape: {shap_2d.shape}")
    # print(shap_2d)

    # Original (likely shape: (num_samples, 64, 1) due to univariate input)
    data_np = X_train_tensor.detach().cpu().numpy()
    print(f"Original data_np.shape: {data_np.shape}")  # e.g., (N, 64, 1)
    
    # Convert to strict 2D: (num_samples, timesteps) = (N, 64)
    data_np = np.squeeze(data_np)                   # Removes any trailing singleton dims (e.g., the feature=1)
    
    # Safety guard: if somehow still >2D (rare), flatten all but sample dim
    if data_np.ndim > 2:
        data_np = data_np.reshape(data_np.shape[0], -1)
    
    print(f"Converted data_np.shape: {data_np.shape}")

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
    shap_df.to_csv(f"{directory}shap_values.csv", index=False)
    print("\nSHAP values saved to CSV successfully!")

    # Set the max_display based on feature count
    max_display = min(64, X_train_tensor.shape[1])

    # Create and save the SHAP bar plot
    fig_bar = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.xlabel("Mean |SHAP value|", fontsize=23)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)
    fig_bar.savefig(f"{directory}shap_bar.png", dpi=600, bbox_inches='tight')
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
    # fig_summary_plot.savefig(f"{directory}shap_summary_plot.png", dpi=600, bbox_inches='tight')
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
    fig_violin.savefig(f"{directory}shap_violin.png", dpi=600, bbox_inches='tight')
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
    fig_beeswarm.savefig(f"{directory}shap_beeswarm.png", dpi=600, bbox_inches='tight')
    plt.close(fig_beeswarm)

    print("\nSHAP plots saved successfully!")
    
    return


# execution 
if __name__ == "__main__":
    
    # create export folder to store outputs
    # e.g. create folder "../run/exp"
    def create_directory(base_path = "../run/bigru/before_boruta/"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()

    # define columns in model training
    cols = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
            'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
            'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 
            'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39',
            'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 
            'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 
            'A60', 'A61', 'A62', 'A63']

    # load data: train, val and test
    for datatype in['training', 'validation', 'test']:
        if datatype == 'training':
            train_path = "../data/train.csv"  
            X_train, y_train = prepare_df(train_path, cols, datatype, directory)
            plot_target_histograms(y_train, datatype, directory)
            X_train_scaled, y_train_scaled,  feature_scaler, target_scaler = standard_df(X_train, y_train, directory)
            # display(X_train_scaled[:1])
            # display(y_train_scaled[:1])

        elif datatype == "validation":
            val_path = "../data/val.csv" 
            X_val, y_val = prepare_df(val_path, cols, datatype, directory)
            plot_target_histograms(y_val, datatype, directory)
            X_val_scaled = feature_scaler.transform(X_val)
            y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
            
        else:
            test_path = "../data/test.csv" 
            X_test, y_test = prepare_df(test_path, cols, datatype, directory)
            plot_target_histograms(y_test, datatype, directory)
            X_test_scaled = feature_scaler.transform(X_test)
            y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
    print(f"\n⚡X_train.head(1): \n{X_train.head(1)}")

    # --- Configuration (Non-tunable) ---
    INPUT_SIZE = 1 # Number of input features
    OUTPUT_SIZE = 1  # Number of output neurons 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔋 Model was trained via {DEVICE}")


    # to tensor
    train_tensor = prepare_tensor(X_train_scaled, y_train_scaled)
    X_train_tensor, y_train_tensor = train_tensor.tensors  # tensors is a tuple (X, y)
    print("\n⚡X_train_tensor shape:", X_train_tensor.shape)
    print(f'X_train_tensor[:1]: \n{X_train_tensor[:1]}')
    
    val_tensor = prepare_tensor(X_val_scaled, y_val_scaled)
    test_tensor = prepare_tensor(X_test_scaled, y_test_scaled)        
            
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
    study.optimize(partial(objective, save_dir=directory), n_trials=2000)
    
    print("\nOptimization Complete.")
    print(f"Best Value (MSE): {study.best_value:.4f}")
    print("Best Hyperparameters:", study.best_params)

    #end_time
    endtime = datetime.now()
    used_time = endtime - starttime
    
    print(f"\n--xgb training time--: {used_time }")


    # retrain the model
    final_model = retrain(study, INPUT_SIZE, OUTPUT_SIZE, train_tensor, val_tensor, directory)

    # save train, val and testtest scatter plot
    for tipe in['train', 'val', 'test']:
        if tipe == 'train':
            data_tensor = train_tensor
        elif tipe == 'val':
            data_tensor = val_tensor
        else:  ## type=='test'
            data_tensor = test_tensor
        evaluate_and_plot(final_model, data_tensor, feature_scaler, target_scaler, tipe, directory)

    # optuna history
    plot_optuna_results(study, directory)

    # compute SHAP
    compute_and_plot_shap(final_model, X_train_tensor, cols, directory)


    # compute SHAP
    compute_and_plot_shap(final_model, X_train_tensor, cols, directory)