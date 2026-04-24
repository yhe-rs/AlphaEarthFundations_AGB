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


try:
    gpus = GPUtil.getGPUs()

    if not gpus:
        print("\nNo GPU detected.")
    else:
        for i, gpu in enumerate(gpus):
            print(f"\n******GPU {i + 1} Information:")
            print(f"ID: {gpu.id}")
            print(f"Name: {gpu.name}")
            print(f"Driver: {gpu.driver}")
            print(f"GPU Memory Total: {gpu.memoryTotal} MB")
            print(f"GPU Memory Free: {gpu.memoryFree} MB")
            print(f"GPU Memory Used: {gpu.memoryUsed} MB")
            print(f"GPU Load: {gpu.load * 100}%")
            print(f"GPU Temperature: {gpu.temperature}°C")

except Exception:
    print("\nNo GPU detected or NVIDIA driver unavailable.")


from eBoruta import eBoruta
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import time
from glob import glob
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import plotly.io as pio
import shap
import datetime
import math

from plotly.io import show
from optuna.importance import get_param_importances, MeanDecreaseImpurityImportanceEvaluator
from datetime import datetime
# display(HTML("<style>.container { width:80% !important; }</style>"))
# pd.set_option("display.max_colwidth", 100)

print("\n******Current working dir", os.getcwd())
import torch
import optuna
# import optunahub

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
# from xgboost import dask as dxgb
# from xgboost.dask import DaskDMatrix
# from xgboost.dask import DaskQuantileDMatrix

print('xgb=:',xgb.__version__)
print('optuna=:',optuna.__version__)
# print('optunahub=:',optunahub.__version__)

# load and preprocess data
def prepare_DMatrix_array(csv_path, cols, dataset, save_dir, cover_type=None):
    """
    Load data from a CSV file, preprocess it, and convert it into a DMatrix format for XGBoost.

    Parameters:
    - csv_path: Path to the CSV file.
    - col: List of column names for features.
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X_train: DataFrame of features.
    - y_train: Series of target variable.
    - dtrain: DMatrix for XGBoost.
    """
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Filter by land cover if specified
    if cover_type is not None:
        df = df[df["Cover"] == cover_type]
        print(f"Filtering for cover type: {cover_type}")
    else:
        print("Using all land cover types")

    # Separate features and target variable for training and testing
    X = df[cols]
    y = df["AGBD"]
    
    print(f"\nTotal {dataset} samples=:", len(X))
    
    # Converting the datasets into DMatrix format for XGBoost
    dMatrix = xgb.DMatrix(X, label=y)

    print("\nxgb.DMatrix transition successfully!")
    with open(f"{save_dir}_used_features.txt", "w") as file:
         file.write(f"used features: \n{cols}")
        
    return X, y, dMatrix




# plot hist of target via in training dataset
def plot_target_histograms(y_train, dataset, save_dir):
    """
    Plot a histogram of the target variable in the training dataset and save it as an image file.

    Parameters:
    - y_train: Series of target variable.
    - save_dir: Directory path to save the plot.

    Returns:
    - None
    """
    
    # Calculate the sample counts
    train_count = len(y_train)
    min_value = y_train.min()
    max_value = y_train.max()

    plt.figure(figsize=(4, 3))

    # Plot histogram for training data
    plt.hist(y_train, bins=200, alpha=0.7, 
             label=f'Total samples (n={train_count}) \nMin value={min_value:.3f} Mg/ha \nMax value={max_value:.3f} Mg/ha', 
             color='blue')

    plt.xlabel('AGB (Mg/ha)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {dataset} set')
    plt.legend(loc='upper right', frameon=False)

    plt.savefig(f"{save_dir}hist-{dataset}.png", dpi=600, pad_inches=0.02, bbox_inches='tight')
    # plt.show(block=False)
    plt.close()

    print(f"\n⚡ Histogram of {dataset} set saved successfully!")
    return




# define objective func
def objective(trial, save_dir):

    """
    Perform hyperparameter tuning for an XGBoost model using Optuna.

    Parameters:
    - trial: Optuna trial object.
    - save_dir: Directory path to save results.

    Returns:
    - The mean RMSE of the best model.
    """
    
    param = {# paras
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method' : "hist", 
        'device' : "cpu",
        'booster': "gbtree",
        'seed': 42, 
        # 'booster': trial.suggest_categorical("booster", ["dart"]),
        
        # 'learning_rate': 0.291,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, step=0.001, log=False), #range: [0,1]
        'gamma': trial.suggest_float('gamma',0, 20, step=0.01, log=False), #range:  [0, ]
        'max_depth': trial.suggest_int('max_depth', 3, 30), #range:  [0, ]
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), #range:  [0, ]
        'subsample': trial.suggest_float('subsample', 0.4, 1.0,  step=0.001, log=False), #range:  (0,1]
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0,  step=0.001, log=False),#range:   (0, 1]
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 30, log=False,step=0.001),  # L2 regularization range: [0, ]
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 30, log=False,step=0.001),     # L1 regularization range: [0, ]
    }
    

    if param["booster"] == "dart":
        # param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        # param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = 0.5#trial.suggest_float("rate_drop", 1e-8, 1.0, log=False)
        param["skip_drop"] = 0.5#trial.suggest_float("skip_drop", 1e-8, 1.0, log=False)


    # evals_result = {}
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-rmse')        
    # train
    results = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=0, # change to 2 to view train-rmse valid-rmse per boost round,  printed every 2 boosting stages
        callbacks=[pruning_callback],
        # evals_result=evals_result 
    )
    # print(evals_result)
        
    trial.set_user_attr('best_iteration', results.best_iteration)

    # print the best round per trial
    print(f"⚠️ Trial {trial.number} completed at index: {results.best_iteration}\n")
    return results.best_score



def get_best_trial_metrics(study, dtrain, save_dir):
    """
    Evaluate the best trial from an Optuna study and calculate its metric

    Parameters:
    - study: Optuna study object.
    - dtrain: XGBoost DMatrix for training.
    - save_dir: Directory path to save results.

    Returns:
    - A dictionary containing the best RMSE and R2 values.
    """

    param = study.best_trial.params.copy()
    param['seed'] = 42
    best_iteration = study.best_trial.user_attrs['best_iteration']
    print(f"⚠️  best_iteration (index) is:{best_iteration}")

    # Make sure *rmse* is the official eval metric – do NOT add `r2`
    eval_metric = param.get("eval_metric", [])
    if isinstance(eval_metric, str):
        eval_metric = [eval_metric]
    if "rmse" not in eval_metric:
        eval_metric.append("rmse")
    # IMPORTANT: do **not** append "r2" – it is unknown in this build
    param["eval_metric"] = eval_metric


    def r2_eval(preds, dmat):
        y_true = dmat.get_label()
        ss_res = np.sum((y_true - preds) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        return "r2", r2     # (name, value)

    # empty dictionary to store the results# 
    evals_result = {}
    results = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=best_iteration+1,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        custom_metric=r2_eval,           # <- custom metric
        evals_result=evals_result,
        verbose_eval=10, # change to 2 to view train-rmse valid-rmse per boost round,  printed every 2 boosting stages
    )

    # print(evals_result)
    # Prepare the data for the DataFrame by extracting the lists
    train_rmse_history = evals_result['train']['rmse']
    valid_rmse_history = evals_result['valid']['rmse']
    train_r2_history = evals_result['train']['r2']
    valid_r2_history = evals_result['valid']['r2']
    
    # Create and save the DataFrame...
    df = pd.DataFrame({
        'boosting_round': range(1, len(train_rmse_history) + 1),
        'train_rmse': train_rmse_history,
        'valid_rmse': valid_rmse_history,
        'train_r2': train_r2_history,
        'valid_r2': valid_r2_history
    })
    csv_path = os.path.join(save_dir, 'best_trial_train_val_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Exported training metrics to {csv_path}") # Changed message to be more accurate
    
    # --- THE FIX IS HERE ---
    # Get the final score from the end of the history list
    best_rmse = valid_rmse_history[-1]
    best_r2   = valid_r2_history[-1]
    
    # Also, your print statement is misleading since this is not CV
    print(f"\nFinal Validation Metrics: RMSE={best_rmse:.4f}, R2={best_r2:.4f}")
    
    return {"rmse": best_rmse, "r2": best_r2}

    

# export trained model and the best paras
def save_best_results_and_train_model(study, X_train, y_train, save_dir):
    """
    Print the best hyperparameters and score, save results to a text file, and train the final XGBoost model.

    Parameters:
    study: Optuna study object containing the results of hyperparameter optimization
    X_train (array-like): Training features
    y_train (array-like): Training labels
    save_dir (str): Directory to save output files

    Returns:
    The trained XGBoost model
    """
    
    # Print the best hyperparameters and the best score
    print("\nBest trial number:", study.best_trial.number)
    print("Best hyperparameters:", study.best_params)
    print("Best num_boost_round:", study.best_trial.user_attrs['best_iteration']+1)
    print("Best RMSE:", study.best_value)
    print("XGB training time:", used_time)

    # Save the best results to a txt file
    with open(f"{save_dir}_best_para_result.txt", "w") as file:
        file.write(f"Best trial number: {study.best_trial.number}\n")
        file.write(f"Best hyperparameters: {study.best_params}\n")
        file.write(f"Best num_boost_round: {study.best_trial.user_attrs['best_iteration']+1}\n")
        file.write(f"Best RMSE: {study.best_value}\n")
        file.write(f"XGB training time: {used_time}")

    # Converting the datasets into DMatrix format for XGBoost prediction
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Get best parameters from the study
    best_params = study.best_trial.params.copy()
    best_params['seed'] = 42

    # Train the final model
    final_model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=study.best_trial.user_attrs['best_iteration']+1,
        verbose_eval=0, # change to 2 to view train-rmse valid-rmse per boost round, printed every 2 boosting stages
    )

    # Save the final model (optional)
    final_model.save_model(f"{save_dir}final_model_.json")

    print("\nFinal model trained, best parameters saved successfully!")
    return final_model



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
def evaluate_and_plot(final_model, y, x, tipe, save_dir):
    """
    Evaluate the model performance and create a scatter plot with regression line.
    Parameters:
    final_model: Trained model for predictions
    x (array-like): Test dataset for predictions
    y (array-like): Actual values for testing
    tipe: str, 'train' or 'test'
    save_dir (str): Directory to save the output files
    """
    
    # Calculate predictions
    y_pred = final_model.predict(x)

    # Calculate R2 and RMSE
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    rrmse = relative_rmse(y, y_pred)


    # Create the scatter plot with regression line
    figsize = (2 ,2)  # Adjust the figure size as needed
    scale_factor = figsize[0]
    
    plt.figure(figsize=figsize)
    sns.regplot(x=y, 
                y=y_pred, 
                marker='o',
                scatter_kws={'color': '#005AB5', 's': 2.0, 'alpha': 0.6}, 
                line_kws={'color': '#0C7BDC'})

    # Add labels and title
    plt.xlabel("Measured AGB (Mg/ha)", fontsize=4* scale_factor)
    plt.ylabel("Estimated AGB (Mg/ha)", fontsize=4* scale_factor)


    # Find the maximum value
    max_value = max(max(y), max(y_pred))
    
    print("\nmax of y：",max(y))
    print("max of y_pred：",max(y_pred))
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
    slope, intercept = np.polyfit(y, y_pred, 1)

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
    results_df = pd.DataFrame({'Measured_Biomass': y, 'Predicted_Biomass': y_pred})

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




# SHAP based feature importance 
def compute_and_plot_shap(final_model, X_train, save_dir):
    """
    Compute and plot SHAP values with automatic GPU/CPU selection.
    
    Parameters:
    final_model: Trained model for SHAP explanation
    X_train: Training data
    save_dir (str): Directory to save the output files
    """
    
    # # Try to use GPUTree explainer, fall back to TreeExplainer if not available
    if torch.cuda.is_available():
        print("\n--- Using GPU SHAP ---")
        explainer = shap.explainers.GPUTree(final_model, X_train)
    else:
        print("\n--- Using CPU TreeExplainer ---")
        explainer = shap.TreeExplainer(final_model)
    # explainer = shap.TreeExplainer(final_model)

    # Compute SHAP values with additivity check disabled
    shap_values_array = explainer.shap_values(X_train, check_additivity=False)
    print(f'n\🚀 shap_values_array.shape: {shap_values_array.shape}')
    # Wrap shap_values_array in a shap.Explanation object
    shap_values = shap.Explanation(values=shap_values_array, 
                                   base_values=explainer.expected_value,
                                   data=X_train,
                                   feature_names=X_train.columns)

    # Convert SHAP values to a DataFrame for saving
    shap_df = pd.DataFrame(shap_values_array, columns=X_train.columns)
    shap_df.to_csv(f"{save_dir}shap_values.csv", index=False)
    print("\nSHAP values saved to CSV successfully!")

    # Set the max_display based on feature count
    max_display = min(100, X_train.shape[1])

    # Create and save the SHAP bar plot
    fig_bar = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.xlabel("Mean |SHAP value|", fontsize=23)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)
    fig_bar.savefig(f"{save_dir}shap_bar.png", dpi=600, bbox_inches='tight')
    plt.close(fig_bar)

    # Create and save the SHAP summary plot
    fig_summary_plot = plt.figure(figsize=(4, max(1, max_display * 0.3)))
    shap.summary_plot(shap_values, max_display=max_display, show=False)
    # Get the colorbar's axis object
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=23)
    cbar.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
    # Set the label for the colorbar with a larger font size
    cbar.set_ylabel("Feature value", fontsize=23,labelpad=-40)
    plt.xlabel('SHAP value (Impact on model output)', fontsize=20)  # Adjust x-axis label size
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=22)    
    fig_summary_plot.savefig(f"{save_dir}shap_summary_plot.png", dpi=600, bbox_inches='tight')
    plt.close(fig_summary_plot)

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

    # Plot intermediate values
    fig_intermediate_values = optuna.visualization.plot_intermediate_values(study)
    fig_intermediate_values.update_layout(
        title="Intermediate Objective Values (RMSE) per Trial",
        xaxis_title="Step (number of rounds or iterations within a trial)",
        yaxis_title="Objective value (RMSE)",
        legend_title="Trial Number",
        # width=width,
        # height=height,        
    )
    fig_intermediate_values.write_html(f"{save_dir}plot_intermediate_values.html")
    #fig_intermediate_values.show(close=True)

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


# execution 
if __name__ == "__main__":
    
    # create export folder to store outputs
    # e.g. create folder "../run/exp"
    def create_directory(base_path = "../run/ssp/xgb/before_boruta/yearly_stats_median/"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()

    # define columns in model training
    cols = ['S1_RVI_max', 'S1_RVI_mean', 'S1_RVI_median', 'S1_RVI_min', 'S1_RVI_range', 'S1_RVI_std', 
            'S1_VH+VV_max', 'S1_VH+VV_mean', 'S1_VH+VV_median', 'S1_VH+VV_min', 'S1_VH+VV_range', 'S1_VH+VV_std', 
            'S1_VH-VV_max', 'S1_VH-VV_mean', 'S1_VH-VV_median', 'S1_VH-VV_min', 'S1_VH-VV_range', 'S1_VH-VV_std', 
            'S1_VH/VV_max', 'S1_VH/VV_mean', 'S1_VH/VV_median', 'S1_VH/VV_min', 'S1_VH/VV_range', 'S1_VH/VV_std', 
            
            'S1_VH_asm_max', 'S1_VH_asm_mean', 'S1_VH_asm_median', 'S1_VH_asm_min', 'S1_VH_asm_range', 'S1_VH_asm_std', 
            'S1_VH_con_max', 'S1_VH_con_mean', 'S1_VH_con_median', 'S1_VH_con_min', 'S1_VH_con_range', 'S1_VH_con_std', 
            'S1_VH_corr_max', 'S1_VH_corr_mean', 'S1_VH_corr_median', 'S1_VH_corr_min', 'S1_VH_corr_range', 'S1_VH_corr_std', 
            'S1_VH_dent_max', 'S1_VH_dent_mean', 'S1_VH_dent_median', 'S1_VH_dent_min', 'S1_VH_dent_range', 'S1_VH_dent_std', 
            'S1_VH_diss_max', 'S1_VH_diss_mean', 'S1_VH_diss_median', 'S1_VH_diss_min', 'S1_VH_diss_range', 'S1_VH_diss_std', 
            'S1_VH_dvar_max', 'S1_VH_dvar_mean', 'S1_VH_dvar_median', 'S1_VH_dvar_min', 'S1_VH_dvar_range', 'S1_VH_dvar_std', 
            'S1_VH_ent_max', 'S1_VH_ent_mean', 'S1_VH_ent_median', 'S1_VH_ent_min', 'S1_VH_ent_range', 'S1_VH_ent_std', 
            'S1_VH_homo_max', 'S1_VH_homo_mean', 'S1_VH_homo_median', 'S1_VH_homo_min', 'S1_VH_homo_range', 'S1_VH_homo_std', 
            'S1_VH_imcorr1_max', 'S1_VH_imcorr1_mean', 'S1_VH_imcorr1_median', 'S1_VH_imcorr1_min', 'S1_VH_imcorr1_range', 'S1_VH_imcorr1_std', 
            'S1_VH_imcorr2_max', 'S1_VH_imcorr2_mean', 'S1_VH_imcorr2_median', 'S1_VH_imcorr2_min', 'S1_VH_imcorr2_range', 'S1_VH_imcorr2_std', 
            'S1_VH_inertia_max', 'S1_VH_inertia_mean', 'S1_VH_inertia_median', 'S1_VH_inertia_min', 'S1_VH_inertia_range', 'S1_VH_inertia_std', 
            'S1_VH_prom_max', 'S1_VH_prom_mean', 'S1_VH_prom_median', 'S1_VH_prom_min', 'S1_VH_prom_range', 'S1_VH_prom_std', 
            'S1_VH_savg_max', 'S1_VH_savg_mean', 'S1_VH_savg_median', 'S1_VH_savg_min', 'S1_VH_savg_range', 'S1_VH_savg_std', 
            'S1_VH_sent_max', 'S1_VH_sent_mean', 'S1_VH_sent_median', 'S1_VH_sent_min', 'S1_VH_sent_range', 'S1_VH_sent_std', 
            'S1_VH_shade_max', 'S1_VH_shade_mean', 'S1_VH_shade_median', 'S1_VH_shade_min', 'S1_VH_shade_range', 'S1_VH_shade_std', 
            'S1_VH_svar_max', 'S1_VH_svar_mean', 'S1_VH_svar_median', 'S1_VH_svar_min', 'S1_VH_svar_range', 'S1_VH_svar_std', 
            'S1_VH_var_max', 'S1_VH_var_mean', 'S1_VH_var_median', 'S1_VH_var_min', 'S1_VH_var_range', 'S1_VH_var_std', 
            'S1_VH_max', 'S1_VH_mean', 'S1_VH_median', 'S1_VH_min', 'S1_VH_range', 'S1_VH_std', 
            'S1_VV_asm_max', 'S1_VV_asm_mean', 'S1_VV_asm_median', 'S1_VV_asm_min', 'S1_VV_asm_range', 'S1_VV_asm_std', 
            'S1_VV_con_max', 'S1_VV_con_mean', 'S1_VV_con_median', 'S1_VV_con_min', 'S1_VV_con_range', 'S1_VV_con_std', 
            'S1_VV_corr_max', 'S1_VV_corr_mean', 'S1_VV_corr_median', 'S1_VV_corr_min', 'S1_VV_corr_range', 'S1_VV_corr_std', 
            'S1_VV_dent_max', 'S1_VV_dent_mean', 'S1_VV_dent_median', 'S1_VV_dent_min', 'S1_VV_dent_range', 'S1_VV_dent_std', 
            'S1_VV_diss_max', 'S1_VV_diss_mean', 'S1_VV_diss_median', 'S1_VV_diss_min', 'S1_VV_diss_range', 'S1_VV_diss_std', 
            'S1_VV_dvar_max', 'S1_VV_dvar_mean', 'S1_VV_dvar_median', 'S1_VV_dvar_min', 'S1_VV_dvar_range', 'S1_VV_dvar_std', 
            'S1_VV_ent_max', 'S1_VV_ent_mean', 'S1_VV_ent_median', 'S1_VV_ent_min', 'S1_VV_ent_range', 'S1_VV_ent_std', 
            'S1_VV_homo_max', 'S1_VV_homo_mean', 'S1_VV_homo_median', 'S1_VV_homo_min', 'S1_VV_homo_range', 'S1_VV_homo_std', 
            'S1_VV_imcorr1_max', 'S1_VV_imcorr1_mean', 'S1_VV_imcorr1_median', 'S1_VV_imcorr1_min', 'S1_VV_imcorr1_range', 'S1_VV_imcorr1_std', 
            'S1_VV_imcorr2_max', 'S1_VV_imcorr2_mean', 'S1_VV_imcorr2_median', 'S1_VV_imcorr2_min', 'S1_VV_imcorr2_range', 'S1_VV_imcorr2_std', 
            'S1_VV_inertia_max', 'S1_VV_inertia_mean', 'S1_VV_inertia_median', 'S1_VV_inertia_min', 'S1_VV_inertia_range', 'S1_VV_inertia_std', 
            'S1_VV_prom_max', 'S1_VV_prom_mean', 'S1_VV_prom_median', 'S1_VV_prom_min', 'S1_VV_prom_range', 'S1_VV_prom_std', 
            'S1_VV_savg_max', 'S1_VV_savg_mean', 'S1_VV_savg_median', 'S1_VV_savg_min', 'S1_VV_savg_range', 'S1_VV_savg_std', 
            'S1_VV_sent_max', 'S1_VV_sent_mean', 'S1_VV_sent_median', 'S1_VV_sent_min', 'S1_VV_sent_range', 'S1_VV_sent_std', 
            'S1_VV_shade_max', 'S1_VV_shade_mean', 'S1_VV_shade_median', 'S1_VV_shade_min', 'S1_VV_shade_range', 'S1_VV_shade_std', 
            'S1_VV_svar_max', 'S1_VV_svar_mean', 'S1_VV_svar_median', 'S1_VV_svar_min', 'S1_VV_svar_range', 'S1_VV_svar_std', 
            'S1_VV_var_max', 'S1_VV_var_mean', 'S1_VV_var_median', 'S1_VV_var_min', 'S1_VV_var_range', 'S1_VV_var_std', 
            'S1_VV_max', 'S1_VV_mean', 'S1_VV_median', 'S1_VV_min', 'S1_VV_range', 'S1_VV_std', 

            
            'S2_B2_max', 'S2_B2_mean', 'S2_B2_median', 'S2_B2_min', 'S2_B2_range', 'S2_B2_std', 
            'S2_B3_max', 'S2_B3_mean', 'S2_B3_median', 'S2_B3_min', 'S2_B3_range', 'S2_B3_std', 
            'S2_B4_max', 'S2_B4_mean', 'S2_B4_median', 'S2_B4_min', 'S2_B4_range', 'S2_B4_std', 
            'S2_B5_max', 'S2_B5_mean', 'S2_B5_median', 'S2_B5_min', 'S2_B5_range', 'S2_B5_std', 
            'S2_B6_max', 'S2_B6_mean', 'S2_B6_median', 'S2_B6_min', 'S2_B6_range', 'S2_B6_std', 
            'S2_B7_max', 'S2_B7_mean', 'S2_B7_median', 'S2_B7_min', 'S2_B7_range', 'S2_B7_std', 
            'S2_B8A_max', 'S2_B8A_mean', 'S2_B8A_median', 'S2_B8A_min', 'S2_B8A_range', 'S2_B8A_std', 
            'S2_B8_max', 'S2_B8_mean', 'S2_B8_median', 'S2_B8_min', 'S2_B8_range', 'S2_B8_std', 
            'S2_B11_max', 'S2_B11_mean', 'S2_B11_median', 'S2_B11_min', 'S2_B11_range', 'S2_B11_std', 
            'S2_B12_max', 'S2_B12_mean', 'S2_B12_median', 'S2_B12_min', 'S2_B12_range', 'S2_B12_std', 
            
            'S2_CIgreen_max', 'S2_CIgreen_mean', 'S2_CIgreen_median', 'S2_CIgreen_min', 'S2_CIgreen_range', 'S2_CIgreen_std', 
            'S2_CIre_max', 'S2_CIre_mean', 'S2_CIre_median', 'S2_CIre_min', 'S2_CIre_range', 'S2_CIre_std', 
            'S2_DVI_max', 'S2_DVI_mean', 'S2_DVI_median', 'S2_DVI_min', 'S2_DVI_range', 'S2_DVI_std', 
            'S2_EVI1_max', 'S2_EVI1_mean', 'S2_EVI1_median', 'S2_EVI1_min', 'S2_EVI1_range', 'S2_EVI1_std', 
            'S2_EVI2_max', 'S2_EVI2_mean', 'S2_EVI2_median', 'S2_EVI2_min', 'S2_EVI2_range', 'S2_EVI2_std', 
            'S2_EVIre1_max', 'S2_EVIre1_mean', 'S2_EVIre1_median', 'S2_EVIre1_min', 'S2_EVIre1_range', 'S2_EVIre1_std', 
            'S2_EVIre2_max', 'S2_EVIre2_mean', 'S2_EVIre2_median', 'S2_EVIre2_min', 'S2_EVIre2_range', 'S2_EVIre2_std', 
            'S2_EVIre3_max', 'S2_EVIre3_mean', 'S2_EVIre3_median', 'S2_EVIre3_min', 'S2_EVIre3_range', 'S2_EVIre3_std', 
            'S2_GNDVI_max', 'S2_GNDVI_mean', 'S2_GNDVI_median', 'S2_GNDVI_min', 'S2_GNDVI_range', 'S2_GNDVI_std', 
            'S2_IRECI_max', 'S2_IRECI_mean', 'S2_IRECI_median', 'S2_IRECI_min', 'S2_IRECI_range', 'S2_IRECI_std', 
            'S2_MCARI1_max', 'S2_MCARI1_mean', 'S2_MCARI1_median', 'S2_MCARI1_min', 'S2_MCARI1_range', 'S2_MCARI1_std', 
            'S2_MCARI2_max', 'S2_MCARI2_mean', 'S2_MCARI2_median', 'S2_MCARI2_min', 'S2_MCARI2_range', 'S2_MCARI2_std', 
            'S2_MCARI3_max', 'S2_MCARI3_mean', 'S2_MCARI3_median', 'S2_MCARI3_min', 'S2_MCARI3_range', 'S2_MCARI3_std', 
            'S2_MTCI1_max', 'S2_MTCI1_mean', 'S2_MTCI1_median', 'S2_MTCI1_min', 'S2_MTCI1_range', 'S2_MTCI1_std', 
            'S2_MTCI2_max', 'S2_MTCI2_mean', 'S2_MTCI2_median', 'S2_MTCI2_min', 'S2_MTCI2_range', 'S2_MTCI2_std', 
            'S2_MTCI3_max', 'S2_MTCI3_mean', 'S2_MTCI3_median', 'S2_MTCI3_min', 'S2_MTCI3_range', 'S2_MTCI3_std', 
            'S2_NDI45_max', 'S2_NDI45_mean', 'S2_NDI45_median', 'S2_NDI45_min', 'S2_NDI45_range', 'S2_NDI45_std', 
            'S2_NDRE1_max', 'S2_NDRE1_mean', 'S2_NDRE1_median', 'S2_NDRE1_min', 'S2_NDRE1_range', 'S2_NDRE1_std', 
            'S2_NDRE2_max', 'S2_NDRE2_mean', 'S2_NDRE2_median', 'S2_NDRE2_min', 'S2_NDRE2_range', 'S2_NDRE2_std', 
            'S2_NDRE3_max', 'S2_NDRE3_mean', 'S2_NDRE3_median', 'S2_NDRE3_min', 'S2_NDRE3_range', 'S2_NDRE3_std', 
            'S2_NDVI56_max', 'S2_NDVI56_mean', 'S2_NDVI56_median', 'S2_NDVI56_min', 'S2_NDVI56_range', 'S2_NDVI56_std', 
            'S2_NDVI57_max', 'S2_NDVI57_mean', 'S2_NDVI57_median', 'S2_NDVI57_min', 'S2_NDVI57_range', 'S2_NDVI57_std', 
            'S2_NDVI68a_max', 'S2_NDVI68a_mean', 'S2_NDVI68a_median', 'S2_NDVI68a_min', 'S2_NDVI68a_range', 'S2_NDVI68a_std', 
            'S2_NDVI78a_max', 'S2_NDVI78a_mean', 'S2_NDVI78a_median', 'S2_NDVI78a_min', 'S2_NDVI78a_range', 'S2_NDVI78a_std', 
            'S2_NDWI1_max', 'S2_NDWI1_mean', 'S2_NDWI1_median', 'S2_NDWI1_min', 'S2_NDWI1_range', 'S2_NDWI1_std', 
            'S2_NDWI2_max', 'S2_NDWI2_mean', 'S2_NDWI2_median', 'S2_NDWI2_min', 'S2_NDWI2_range', 'S2_NDWI2_std', 
            'S2_NIRv_max', 'S2_NIRv_mean', 'S2_NIRv_median', 'S2_NIRv_min', 'S2_NIRv_range', 'S2_NIRv_std', 
            'S2_NLI_max', 'S2_NLI_mean', 'S2_NLI_median', 'S2_NLI_min', 'S2_NLI_range', 'S2_NLI_std', 
            'S2_OSAVI_max', 'S2_OSAVI_mean', 'S2_OSAVI_median', 'S2_OSAVI_min', 'S2_OSAVI_range', 'S2_OSAVI_std',
            'S2_PSSRa_max', 'S2_PSSRa_mean', 'S2_PSSRa_median', 'S2_PSSRa_min', 'S2_PSSRa_range', 'S2_PSSRa_std', 
            'S2_SAVI_max', 'S2_SAVI_mean', 'S2_SAVI_median', 'S2_SAVI_min', 'S2_SAVI_range', 'S2_SAVI_std', 
            'S2_SR_max', 'S2_SR_mean', 'S2_SR_median', 'S2_SR_min', 'S2_SR_range', 'S2_SR_std', 
            'S2_kNDVI_max', 'S2_kNDVI_mean', 'S2_kNDVI_median', 'S2_kNDVI_min', 'S2_kNDVI_range', 'S2_kNDVI_std', 
            
            'P2_HH+HV_max', 'P2_HH+HV_mean', 'P2_HH+HV_median', 'P2_HH+HV_min', 'P2_HH+HV_range', 'P2_HH+HV_std', 
            'P2_HH-HV_max', 'P2_HH-HV_mean', 'P2_HH-HV_median', 'P2_HH-HV_min', 'P2_HH-HV_range', 'P2_HH-HV_std', 
            'P2_HH_asm_max', 'P2_HH_asm_mean', 'P2_HH_asm_median', 'P2_HH_asm_min', 'P2_HH_asm_range', 'P2_HH_asm_std', 
            'P2_HH_con_max', 'P2_HH_con_mean', 'P2_HH_con_median', 'P2_HH_con_min', 'P2_HH_con_range', 'P2_HH_con_std', 
            'P2_HH_corr_max', 'P2_HH_corr_mean', 'P2_HH_corr_median', 'P2_HH_corr_min', 'P2_HH_corr_range', 'P2_HH_corr_std', 
            'P2_HH_dent_max', 'P2_HH_dent_mean', 'P2_HH_dent_median', 'P2_HH_dent_min', 'P2_HH_dent_range', 'P2_HH_dent_std', 
            'P2_HH_diss_max', 'P2_HH_diss_mean', 'P2_HH_diss_median', 'P2_HH_diss_min', 'P2_HH_diss_range', 'P2_HH_diss_std', 
            'P2_HH_dvar_max', 'P2_HH_dvar_mean', 'P2_HH_dvar_median', 'P2_HH_dvar_min', 'P2_HH_dvar_range', 'P2_HH_dvar_std', 
            'P2_HH_ent_max', 'P2_HH_ent_mean', 'P2_HH_ent_median', 'P2_HH_ent_min', 'P2_HH_ent_range', 'P2_HH_ent_std', 
            'P2_HH_homo_max', 'P2_HH_homo_mean', 'P2_HH_homo_median', 'P2_HH_homo_min', 'P2_HH_homo_range', 'P2_HH_homo_std',
            'P2_HH_imcorr1_max', 'P2_HH_imcorr1_mean', 'P2_HH_imcorr1_median', 'P2_HH_imcorr1_min', 'P2_HH_imcorr1_range', 'P2_HH_imcorr1_std', 
            'P2_HH_imcorr2_max', 'P2_HH_imcorr2_mean', 'P2_HH_imcorr2_median', 'P2_HH_imcorr2_min', 'P2_HH_imcorr2_range', 'P2_HH_imcorr2_std', 
            'P2_HH_inertia_max', 'P2_HH_inertia_mean', 'P2_HH_inertia_median', 'P2_HH_inertia_min', 'P2_HH_inertia_range', 'P2_HH_inertia_std', 
            'P2_HH_prom_max', 'P2_HH_prom_mean', 'P2_HH_prom_median', 'P2_HH_prom_min', 'P2_HH_prom_range', 'P2_HH_prom_std', 
            'P2_HH_savg_max', 'P2_HH_savg_mean', 'P2_HH_savg_median', 'P2_HH_savg_min', 'P2_HH_savg_range', 'P2_HH_savg_std', 
            'P2_HH_sent_max', 'P2_HH_sent_mean', 'P2_HH_sent_median', 'P2_HH_sent_min', 'P2_HH_sent_range', 'P2_HH_sent_std',
            'P2_HH_shade_max', 'P2_HH_shade_mean', 'P2_HH_shade_median', 'P2_HH_shade_min', 'P2_HH_shade_range', 'P2_HH_shade_std', 
            'P2_HH_svar_max', 'P2_HH_svar_mean', 'P2_HH_svar_median', 'P2_HH_svar_min', 'P2_HH_svar_range', 'P2_HH_svar_std', 
            'P2_HH_var_max', 'P2_HH_var_mean', 'P2_HH_var_median', 'P2_HH_var_min', 'P2_HH_var_range', 'P2_HH_var_std', 
            'P2_HH_max', 'P2_HH_mean', 'P2_HH_median', 'P2_HH_min', 'P2_HH_range', 'P2_HH_std',
            'P2_HV/HH_max', 'P2_HV/HH_mean', 'P2_HV/HH_median', 'P2_HV/HH_min', 'P2_HV/HH_range', 'P2_HV/HH_std', 
            'P2_HV_asm_max', 'P2_HV_asm_mean', 'P2_HV_asm_median', 'P2_HV_asm_min', 'P2_HV_asm_range', 'P2_HV_asm_std', 
            'P2_HV_con_max', 'P2_HV_con_mean', 'P2_HV_con_median', 'P2_HV_con_min', 'P2_HV_con_range', 'P2_HV_con_std', 
            'P2_HV_corr_max', 'P2_HV_corr_mean', 'P2_HV_corr_median', 'P2_HV_corr_min', 'P2_HV_corr_range', 'P2_HV_corr_std', 
            'P2_HV_dent_max', 'P2_HV_dent_mean', 'P2_HV_dent_median', 'P2_HV_dent_min', 'P2_HV_dent_range', 'P2_HV_dent_std', 
            'P2_HV_diss_max', 'P2_HV_diss_mean', 'P2_HV_diss_median', 'P2_HV_diss_min', 'P2_HV_diss_range', 'P2_HV_diss_std', 
            'P2_HV_dvar_max', 'P2_HV_dvar_mean', 'P2_HV_dvar_median', 'P2_HV_dvar_min', 'P2_HV_dvar_range', 'P2_HV_dvar_std', 
            'P2_HV_ent_max', 'P2_HV_ent_mean', 'P2_HV_ent_median', 'P2_HV_ent_min', 'P2_HV_ent_range', 'P2_HV_ent_std', 
            'P2_HV_homo_max', 'P2_HV_homo_mean', 'P2_HV_homo_median', 'P2_HV_homo_min', 'P2_HV_homo_range', 'P2_HV_homo_std', 
            'P2_HV_imcorr1_max', 'P2_HV_imcorr1_mean', 'P2_HV_imcorr1_median', 'P2_HV_imcorr1_min', 'P2_HV_imcorr1_range', 'P2_HV_imcorr1_std',
            'P2_HV_imcorr2_max', 'P2_HV_imcorr2_mean', 'P2_HV_imcorr2_median', 'P2_HV_imcorr2_min', 'P2_HV_imcorr2_range', 'P2_HV_imcorr2_std',
            'P2_HV_inertia_max', 'P2_HV_inertia_mean', 'P2_HV_inertia_median', 'P2_HV_inertia_min', 'P2_HV_inertia_range', 'P2_HV_inertia_std', 
            'P2_HV_prom_max', 'P2_HV_prom_mean', 'P2_HV_prom_median', 'P2_HV_prom_min', 'P2_HV_prom_range', 'P2_HV_prom_std', 
            'P2_HV_savg_max', 'P2_HV_savg_mean', 'P2_HV_savg_median', 'P2_HV_savg_min', 'P2_HV_savg_range', 'P2_HV_savg_std', 
            'P2_HV_sent_max', 'P2_HV_sent_mean', 'P2_HV_sent_median', 'P2_HV_sent_min', 'P2_HV_sent_range', 'P2_HV_sent_std', 
            'P2_HV_shade_max', 'P2_HV_shade_mean', 'P2_HV_shade_median', 'P2_HV_shade_min', 'P2_HV_shade_range', 'P2_HV_shade_std', 
            'P2_HV_svar_max', 'P2_HV_svar_mean', 'P2_HV_svar_median', 'P2_HV_svar_min', 'P2_HV_svar_range', 'P2_HV_svar_std', 
            'P2_HV_var_max', 'P2_HV_var_mean', 'P2_HV_var_median', 'P2_HV_var_min', 'P2_HV_var_range', 'P2_HV_var_std', 
            'P2_HV_max', 'P2_HV_mean', 'P2_HV_median', 'P2_HV_min', 'P2_HV_range', 'P2_HV_std', 
            'Aspect', 'Ele', 'Slope'
           ]
    # load data: train, val and test
    for dataset in['training', 'validation', 'test']:
        if dataset == 'training':
            train_path = "../data/train_yearly_stats_median.csv"  
            X_train, y_train, dtrain = prepare_DMatrix_array(train_path, cols, dataset, directory, cover_type=None)
            plot_target_histograms(y_train, dataset, directory)
            # split cover
            X_train_Croplands, y_train_Croplands,dtrain_Croplands  = prepare_DMatrix_array(train_path, cols, dataset, directory, cover_type="Croplands")
            X_train_Forests, y_train_Forests,dtrain_Forests  = prepare_DMatrix_array(train_path, cols, dataset, directory, cover_type="Forests")
            X_train_Savannas, y_train_Savannas,dtrain_Savannas = prepare_DMatrix_array(train_path, cols, dataset, directory, cover_type="Savannas")
            X_train_Shrub_grass_lands, y_train_Shrub_grass_lands,dtrain_Shrub_grass_lands  = prepare_DMatrix_array(train_path, cols, dataset, directory, cover_type="Shrub_grass_lands")
            
        elif dataset == "validation":
            val_path ="../data/val_yearly_stats_median.csv" 
            X_val, y_val, dval = prepare_DMatrix_array(val_path, cols, dataset, directory)
            plot_target_histograms(y_val, dataset, directory)
            
        else:
            test_path = "../data/test_yearly_stats_median.csv" 
            X_test, y_test, dtest = prepare_DMatrix_array(test_path, cols, dataset, directory)
            plot_target_histograms(y_test, dataset, directory)
            # split cover
            X_test_Croplands, y_test_Croplands,dtest_Croplands = prepare_DMatrix_array(test_path, cols, dataset, directory, cover_type="Croplands")
            X_test_Forests, y_testForests,dtest_Forests = prepare_DMatrix_array(test_path, cols, dataset, directory, cover_type="Forests")
            X_test_Savannas, y_test_Savannas,dtest_Savannas= prepare_DMatrix_array(test_path, cols, dataset, directory, cover_type="Savannas")
            X_test_Shrub_grass_lands, y_test_Shrub_grass_lands,dtest_Shrub_grass_lands = prepare_DMatrix_array(test_path, cols, dataset, directory, cover_type="Shrub_grass_lands")  
            
    # training starts
    starttime = datetime.now()

    # # Set up the Optuna study using the TPESampler and MedianPruner
    # study = optuna.create_study(
    #     # sampler=optuna.samplers.RandomSampler(seed=42),
    #     sampler=optuna.samplers.TPESampler(n_startup_trials=100),
    #     # sampler=optunahub.load_module( "samplers/auto_sampler").AutoSampler(n_startup_trials=100), 
    #     # pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=100),
    #     direction='minimize'
    # )
    
    # Set up the Optuna study using the TPESampler and MedianPruner
    study = optuna.create_study(
        # sampler=optuna.samplers.RandomSampler(seed=42),
        sampler=optuna.samplers.TPESampler(n_startup_trials=100),
        # sampler=optuna.samplers.GPSampler(n_startup_trials=100),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=100),
        direction='minimize'
    )
    # optimization
    from functools import partial
    study.optimize(partial(objective, save_dir=directory), 
                   n_trials=2000,
                   n_jobs=64,
                   catch=(xgb.core.XGBoostError,)
                  )
     #end_time
    endtime = datetime.now()
    used_time = endtime - starttime
    
    print(f"\n--xgb training time--: {used_time }")

    # train and test rmse in best trial
    get_best_trial_metrics(study, dtrain, directory)
    
    # save scatterplot of test, prediection in dataframe using trained model
    final_model= save_best_results_and_train_model(study, X_train, y_train, directory)     


    # save train and test scatter plot
    for tipe in ['train_all','train_crop','train_forest','train_savannas','train_shrubgrass',
                 'test_all','test_crop','test_forest','test_savannas','test_shrubgrass']:    
        if tipe == 'train_all':
            y = y_train
            x = dtrain    
        elif tipe == 'train_crop':
            y = y_train_Croplands
            x = dtrain_Croplands   
        elif tipe == 'train_forest':
            y = y_train_Forests
            x = dtrain_Forests    
        elif tipe == 'train_savannas':
            y = y_train_Savannas
            x = dtrain_Savannas    
        elif tipe == 'train_shrubgrass':
            y = y_train_Shrub_grass_lands
            x = dtrain_Shrub_grass_lands    
        elif tipe == 'test_all':
            y = y_test
            x = dtest    
        elif tipe == 'test_crop':
            y = y_test_Croplands
            x = dtest_Croplands    
        elif tipe == 'test_forest':
            y = y_testForests
            x = dtest_Forests    
        elif tipe == 'test_savannas':
            y = y_test_Savannas
            x = dtest_Savannas    
        else:  # test_shrubgrass
            y = y_test_Shrub_grass_lands
            x = dtest_Shrub_grass_lands    
        evaluate_and_plot(final_model, y, x, tipe, directory)
    
    # save SHAP 
    compute_and_plot_shap(final_model, X_train, directory)
    
    # save htlm plots of traing history 
    plot_optuna_results(study, directory)
    print("\nExecution done!")
    
