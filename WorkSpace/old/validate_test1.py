from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np
import matplotlib.pyplot as plt

# 検証するネットワークが保存されたディレクトリ
OBJECT_DIR = r"C:\Users\taniy\Dropbox\_COLLEGE\_Labolatory\_Study\SNN\Neocortex_brian2\cortex\WorkSpace\saved_network\WTA\2024_11_05_14_42_51_WTA - 膜時定数50ms epoch=2_comp"
# OBJECT_DIR = "examined_data/" + "2024_10_21_16_50_48_検証用WTA350ms_comp" + "/"
VALIDATION_NAME = "monitor_50img_heatmap"

PARAMS_PATH = "Brian2_Framework/parameters/WTA/membrane_tau_50/WTA_validate.json"


params = tools.load_parameters(PARAMS_PATH)
np.random.seed(params["seed"])


print("Object directory: ", OBJECT_DIR)
# ===================================== Validationの実行 ==========================================
# weights = np.arange(0, 22.5, 0.5)
# weights = [3.5]
weights = [22]
mean_firing_list = []
for weight in weights:
    params["static_synapse_params_ei"]["w"] = weight
    validator = Validator(
                    target_path=OBJECT_DIR, 
                    assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                    params=params,
                    network_type=params["network_type"],
                    enable_monitor=params["enable_monitor"],
                    labels=params["labels"]
                    )
    if params["enable_monitor"]:    
        path, mean_firing = validator.validate(n_samples=params["n_samples"], examination_name=VALIDATION_NAME + "_" + str(weight))
        mean_firing_list.append(mean_firing)
    else:
        path = validator.validate(n_samples=params["n_samples"], examination_name=VALIDATION_NAME + "_" + str(weight))

if params["enable_monitor"]:
    plt.figure(figsize=(10, 8))
    plt.plot(weights, mean_firing_list, color="black")
    plt.title("Mean firing changes with weight")
    plt.xlabel("weight")
    plt.ylabel("mean firing count")
    plt.xticks(weights, rotation=45)
    plt.yticks(np.arange(0, 102, 2))
    plt.savefig(os.path.join(path, "graphs", "fr_nonzero", "mean_firing_changes.png"))
    plt.close()
