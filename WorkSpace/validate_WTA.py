from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np
import matplotlib.pyplot as plt

# 検証するネットワークが保存されたディレクトリ
OBJECT_DIR = r"C:\Users\taniy\Dropbox\_COLLEGE\_Labolatory\_Study\SNN\Neocortex_brian2\cortex\WorkSpace\saved_network\WTA\2024_11_05_14_42_51_WTA - 膜時定数50ms epoch=2_comp"
VALIDATION_NAME = "TEST"

PARAMS_PATH = "Brian2_Framework/parameters/WTA/_main/WTA_validate.json"


params = tools.load_parameters(PARAMS_PATH)
np.random.seed(params["seed"])


print("Object directory: ", OBJECT_DIR)
# ===================================== Validationの実行 ==========================================

validator = Validator(
                target_path=OBJECT_DIR, 
                assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                params=params,
                network_type=params["network_type"],
                enable_monitor=params["enable_monitor"],
                labels=params["labels"]
                )
validator.validate(n_samples=params["n_samples"], examination_name=VALIDATION_NAME)

