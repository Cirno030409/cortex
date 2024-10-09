from brian2.units import *
import os

from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import json
import numpy as np
OBJECT_DIR = "examined_data/2024_09_07_19_50_59_重みをnpyで保存_comp/" # 重みを保存したディレクトリ

seed = 3
np.random.seed(seed)

PARAMS_PATH = "Brian2_Framework/parameters/WTA/WTA_validate.json"
params = tools.load_parameters(PARAMS_PATH)
#! Neuron & Synapse Parameters
with open(os.path.join(OBJECT_DIR, "parameters_validate.json"), "w") as f:
    json.dump(params, f, indent=4, default=tools.convert_quantity)
validator = Validator(
                    weight_path=f"{OBJECT_DIR}/weights.npy", 
                    assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                    params_json_path=PARAMS_PATH,
                    network_type=params["network_type"])
acc, predict_labels, answer_labels, wronged_image_idx = validator.validate(n_samples=params["n_samples"])
print(f"Accuracy: {acc}")
print(f"Wrongly predicted images: {wronged_image_idx}")

tools.save_parameters(OBJECT_DIR + "parameters_validate.json", params)

# 結果を記録
with open(f"{OBJECT_DIR}/results/result.txt", "w") as f:
    f.write(f"Accuracy: {acc*100}%\n")
    f.write("\n[Answer labels -> Predict labels]\n")
    for i in range(len(answer_labels)):
        f.write(f"Image {i}: {answer_labels[i]} -> {predict_labels[i]}\n")
    f.write("\n[Wrongly predicted images]\n")
    f.write("Wrong Image idx: Answer labels -> Predict labels\n")
    for idx in wronged_image_idx:
        f.write(f"Image {idx}: {answer_labels[idx]} -> {predict_labels[idx]}\n")
        
tools.change_dir_name(OBJECT_DIR, f"_validated_acc={acc*100:.2f}%")