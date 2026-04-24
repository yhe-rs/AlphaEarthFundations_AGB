
import platform
import psutil
import GPUtil

system_info = platform.uname()

print("System Information:")
print(f"System: {system_info.system}")
print(f"Node Name: {system_info.node}")
print(f"Release: {system_info.release}")
print(f"Version: {system_info.version}")
print(f"Machine: {system_info.machine}")
print(f"Processor: {system_info.processor}")

cpu_info = platform.processor()
cpu_count = psutil.cpu_count(logical=False)
logical_cpu_count = psutil.cpu_count(logical=True)

print("\nCPU Information:")
print(f"Processor: {cpu_info}")
print(f"Physical Cores: {cpu_count}")
print(f"Logical Cores: {logical_cpu_count}")

memory_info = psutil.virtual_memory()

print("\nMemory Information:")
print(f"Total Memory: {memory_info.total} bytes")
print(f"Available Memory: {memory_info.available} bytes")
print(f"Used Memory: {memory_info.used} bytes")
print(f"Memory Utilization: {memory_info.percent}%")

disk_info = psutil.disk_usage('/')

print("\nDisk Information:")
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from eBoruta import eBoruta
import pandas as pd
import os
import time

from collections import Counter
from tqdm import tqdm 

def prepare_df(csv_path, cols, save_dir):
    """
    Load data from a CSV file, preprocess it, and convert it into a df for rf.

    Parameters:
    - csv_path: Path to the CSV file.
    - col: List of column names for features.
    - save_dir: Directory path to save preprocessed data.

    Returns:
    - X_train: DataFrame of features.
    - y_train: Series of target variable.
    """
    
    # Load the data
    train_df = pd.read_csv(csv_path)


    # Separate features and target variable for training and testing
    X_train = train_df[cols]
    y_train = train_df["AGBD"]

    X_train.to_csv(f"{save_dir}X_train.csv", index=False)
    y_train.to_csv(f"{save_dir}y_train.csv", index=False)
    
    print(f"\nTotal samples=:", len(X_train))

    print("\nRF dataframe prepared successfully!")
    with open(f"{save_dir}_used_features.txt", "w") as file:
         file.write(f"used features: \n{cols}")
    return X_train, y_train


from collections import Counter

def eboruta_selector(X_train, y_train):
    # Define Regressor with Correct Parameters for GPU Acceleration
    rf_kwargs = {
        'n_estimators':150,
        'max_depth':27,
        'min_samples_split':3,
        'min_samples_leaf':2,
        'max_features':'log2',
        'criterion':'squared_error',
        'random_state':42,
        'n_jobs':-1
                }

    # Initialize the eBoruta Selector with SHAP Approximation and Reduced Iterations
    selector = eBoruta(
        n_iter=2000,
        classification=False,
        percentile=100,
        pvalue=0.01,
        test_size=0,
        test_stratify=False,
        shap=True,
        shap_gpu_tree=True,
        shap_approximate=False,
        shap_check_additivity=False,
        importance_getter=None,
        verbose=1
    )

    selector.fit(X_train, 
                 y_train, 
                 model_type=RandomForestRegressor, # No parentheses here!
                model_init_kwargs=rf_kwargs,       # Parameters go here
                )
    return selector.features_.accepted.tolist()


def run_feature_selection(X_train, y_train, trial, threshold):
    selectors = []
    
    # Run the selector 'trial' times
    for i in tqdm(range(trial)):
        print(f"Running iteration {i+1}")
        selected_features = eboruta_selector(X_train, y_train)
        selectors.extend(selected_features)  # Accumulate selected features
    
    # Count occurrences of each feature
    feature_counts = Counter(selectors)
    
    # Calculate minimum occurrence threshold
    min_occurrences = threshold * trial
    
    # Get features that appear in at least 90% of trials
    frequent_features = [feature for feature, count in feature_counts.items() if count >= min_occurrences]
    
    return frequent_features


# # Example usage
if __name__ == "__main__":
    
    # create export folder to store outputs
    # e.g. create folder "../run/exp"
    def create_directory(base_path = "../run/alpha/rf/boruta/"):
    
        counter = 0
    
        while os.path.exists(base_path + (str(counter) if counter > 0 else "") + "/"):
            counter += 1

        new_directory = base_path + (str(counter) if counter > 0 else "") + "/"
        os.makedirs(new_directory)
        print(f"\n******Created directory: {new_directory}")
    
        return new_directory

    directory = create_directory()

    
    # cols
    cols =[
           'S1_RVI', 'S1_VH+VV', 'S1_VH-VV', 'S1_VH/VV', 
           'S1_VH', 
           'S1_VH_asm', 'S1_VH_con', 'S1_VH_corr', 'S1_VH_dent', 'S1_VH_diss', 'S1_VH_dvar', 'S1_VH_ent', 'S1_VH_homo', 'S1_VH_imcorr1', 'S1_VH_imcorr2', 
           'S1_VH_inertia', 'S1_VH_prom', 'S1_VH_savg', 'S1_VH_sent', 'S1_VH_shade', 'S1_VH_svar', 'S1_VH_var', 
           'S1_VV', 
           'S1_VV_asm', 'S1_VV_con', 'S1_VV_corr', 'S1_VV_dent', 'S1_VV_diss', 'S1_VV_dvar', 'S1_VV_ent', 'S1_VV_homo', 'S1_VV_imcorr1', 'S1_VV_imcorr2', 
           'S1_VV_inertia', 'S1_VV_prom', 'S1_VV_savg', 'S1_VV_sent', 'S1_VV_shade', 'S1_VV_svar', 'S1_VV_var', 
           'S2_B1', 'S2_B10', 'S2_B11', 'S2_B12', 'S2_B2', 'S2_B3', 'S2_B4', 'S2_B5', 'S2_B6', 'S2_B7', 'S2_B8', 'S2_B8A', 'S2_B9', 
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

    csv_path = "../data/train_yearly_median.csv" 
    X_train, y_train= prepare_df(csv_path, cols, save_dir=directory)

    start_time = time.time()
    # 
    num_trial=100
    thld=0.9
    frequent_features = run_feature_selection(X_train, y_train, trial=num_trial, threshold=thld)
    print("Features selected in at least 90% of the trials:", frequent_features)
    
    # End time calculation
    end_time = time.time()
    used_time = end_time - start_time
    # Convert used_time to human-friendly format (HH:MM:SS)
    human_readable_time = time.strftime("%H:%M:%S", time.gmtime(used_time))
    print(f"\n--rf training time--: {human_readable_time}")

    # Save the best results to a txt file
    with open(f"{directory}boruta", "w") as file:
        file.write(f" boruta fts: \n{frequent_features}\n")
        file.write(f" boruta time: {human_readable_time}")           
        file.write(f" num of trial: {num_trial}\n") 
        file.write(f" threshold: {thld}\n")  