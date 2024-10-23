from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np
OBJECT_DIR = "saved_network/" + "2024_09_07_19_50_59_重みをnpyで保存_homesta_comp" + "/"
# OBJECT_DIR = "examined_data/" + "2024_10_21_16_50_48_検証用WTA350ms_comp" + "/" # 重みを保存したディレクトリ
PARAMS_PATH = "Brian2_Framework/parameters/WTA/WTA_validate.json"

seed = 2
np.random.seed(seed)
params = tools.load_parameters(PARAMS_PATH)

print("Object directory: ", OBJECT_DIR)
# ===================================== Validationの実行 ==========================================
validator = Validator(
                    target_path=OBJECT_DIR, 
                    assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                    params_json_path=PARAMS_PATH,
                    network_type=params["network_type"])
validator.validate(n_samples=params["n_samples"])
