from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np

# 検証するネットワークが保存されたディレクトリ
OBJECT_DIR = "saved_network/" + "2024_10_21_13_27_20_検証用WTA150ms_comp" + "/"
# OBJECT_DIR = "examined_data/" + "2024_10_21_16_50_48_検証用WTA350ms_comp" + "/"
VALIDATION_NAME = "電流値計測用 - 膜時定数50ms 10000枚"

PARAMS_PATH = "Brian2_Framework/parameters/WTA/membrane_tau_50/WTA_validate.json"


params = tools.load_parameters(PARAMS_PATH)
np.random.seed(params["seed"])


print("Object directory: ", OBJECT_DIR)
# ===================================== Validationの実行 ==========================================
validator = Validator(
                    target_path=OBJECT_DIR, 
                    assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                    params_json_path=PARAMS_PATH,
                    network_type=params["network_type"],
                    enable_monitor=params["enable_monitor"]   
                    )
validator.validate(n_samples=params["n_samples"], examination_name=VALIDATION_NAME)
