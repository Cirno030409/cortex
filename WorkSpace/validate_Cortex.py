from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np
import matplotlib.pyplot as plt

# 検証するネットワークが保存されたディレクトリ
OBJECT_DIR = r"C:\Users\taniy\Dropbox\_COLLEGE\_Labolatory\_Study\SNN\Neocortex_brian2\cortex\WorkSpace\examined_data\20241125171251_Cortex - mc=5 labels=[0, 1, 2, 3] n=8_初期重み同期_2epoch_comp"

VALIDATION_NAME = "validation"

PARAMS_PATH = "Brian2_Framework/parameters/Cortex/Cortex_validate.json"
PARAMS_MC_PATH = "Brian2_Framework/parameters/Mini_column/Mini_column_validate.json"


params = tools.load_parameters(PARAMS_PATH)
params_mc = tools.load_parameters(PARAMS_MC_PATH)
np.random.seed(params["seed"])


print("Object directory: ", OBJECT_DIR)
# ===================================== Validationの実行 ==========================================

validator = Validator(
                target_path=OBJECT_DIR, 
                assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                params=params,
                params_mc=params_mc,
                network_type=params["network_type"],
                enable_monitor=params["enable_monitor"],
                labels=params["labels"]
                )
validator.validate(n_samples=params["n_samples"], examination_name=VALIDATION_NAME)

