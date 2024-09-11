from Brian2_Framework.Validator import Validator
from brian2.units import *

OBJECT_DIR = "examined_data/2024_09_09_18_54_46_2Epoch回したら？_comp" # 重みを保存したディレクトリ
N_SAMPLES = 10000 # テストデータの数

if __name__ == "__main__":
    seed = 3
    #! Neuron & Synapse Parameters
    neuron_params_e = {
        "I_noise"       : 0,        # 定常入力電流
        "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
        "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
        "taum"          : 100*ms,    # 膜電位の時定数
        "theta_dt"      : 0.05,      # ホメオスタシスの発火閾値の上昇値
        "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
        "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
        "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
        "refractory"    : 2 * ms,   # 不応期
        "v_reset"       : -60,      # リセット電位
        "v_rest"        : -50,      # 静止膜電位
        "v_th"          : -40       # 発火閾値
    }

    neuron_params_i = {
        "I_noise"       : 0,        # 定常入力電流
        "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
        "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
        "taum"          : 10*ms,    # 膜電位の時定数
        "theta_dt"      : 0,      # ホメオスタシスの発火閾値の上昇値
        "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
        "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
        "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
        "refractory"    : 2 * ms,   # 不応期
        "v_reset"       : -60,      # リセット電位
        "v_rest"        : -50,      # 静止膜電位
        "v_th"          : -40       # 発火閾値
    }

    stdp_synapse_params = {
        "wmax": 1,              # 最大重み
        "wmin": 0,              # 最小重み
        "Apre": 1,           # 前ニューロンのスパイクトレースのリセット値
        "Apost": 1,             # 後ニューロンのスパイクトレースのリセット値
        "taupre": 20 * ms,      # 前ニューロンのスパイクトレースの時定数
        "taupost": 20 * ms,     # 後ニューロンのスパイクトレースの時定数
        "nu_pre": 0.01,        # 学習率
        "nu_post": 0.0001,       # 学習率
        "alpha": 0,          # スパイクトレースの収束地点
        "sw": 1,             # 学習の有無の切り替え
    }

    static_synapse_params_ei = {
        "w": 30,
    }

    static_synapse_params_ie = {
        "w": 22,
    }
    validator = Validator(
                        weight_path=f"{OBJECT_DIR}/weights.npy", 
                        assigned_labels_path=f"{OBJECT_DIR}/assigned_labels.pkl", 
                        neuron_params_e=neuron_params_e, neuron_params_i=neuron_params_i, 
                        stdp_synapse_params=stdp_synapse_params, 
                        n_inp=784, n_e=100, n_i=100, max_rate=60, 
                        static_synapse_params_ei=static_synapse_params_ei, static_synapse_params_ie=static_synapse_params_ie,
                        network_type="WTA")
    acc, predict_labels, answer_labels, wronged_image_idx = validator.validate(n_samples=N_SAMPLES)
    print(f"Accuracy: {acc}")
    print(f"Wrongly predicted images: {wronged_image_idx}")
    
    # 結果を記録
    with open(f"{OBJECT_DIR}/result.txt", "w") as f:
        f.write(f"Accuracy: {acc*100}%\n")
        f.write("\n[Answer labels -> Predict labels]\n")
        for i in range(len(answer_labels)):
            f.write(f"Image {i}: {answer_labels[i]} -> {predict_labels[i]}\n")
        f.write("\n[Wrongly predicted images]\n")
        f.write("Wrong Image idx: Answer labels -> Predict labels\n")
        for idx in wronged_image_idx:
            f.write(f"Image {idx}: {answer_labels[idx]} -> {predict_labels[idx]}\n")