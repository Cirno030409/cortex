import matplotlib.pyplot as plt
import Network.Mnist as Mnist
import Network.Plotters as Plotters
from brian2 import *
from Network.Neurons import *
from Network.Synapses import *
import pprint as p
import numpy as np

np.random.seed(14)

neuron_params_e = {
    "I_noise"       : 0,        # 定常入力電流
    "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
    "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
    "taum"          : 10*ms,    # 膜電位の時定数
    "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
    "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
    "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
    "theta_dt"      : 0,        # ホメオスタシスの発火閾値の上昇値
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
    "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
    "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
    "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
    "theta_dt"      : 0,        # ホメオスタシスの発火閾値の上昇値
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
    "nu_pre": 1,        # 学習率
    "nu_post": 1,       # 学習率
    "alpha": 0.01,          # スパイクトレースの収束地点
}

static_synapse_params_ei = {
    "w": 4.
}

static_synapse_params_ie = {
    "w": 2.5,
}

# 要素のインスタンス化
plotter = Plotters.Common_Plotter()
neuron_e = Conductance_LIF(neuron_params_e)
neuron_i = Conductance_LIF(neuron_params_i)
neuron_inp = Poisson_Input()
synapse_e = NonSTDP(static_synapse_params_ei)
synapse_i = NonSTDP(static_synapse_params_ie)
synapse_stdp = STDP(stdp_synapse_params)
plotter = Plotters.Common_Plotter()

# MNISTデータの取得
n_samples = 10
# images, labels = Mnist.get_mnist_image(label=1, n_samples=20, down_sample=1, dataset='train')
images, labels = Mnist.get_mnist_sample(n_samples, "train")
# images[画像枚数][28][28]
# labels[画像枚数]

N = []
S = []

spikemon = {}
statemon_n = {}
statemon_s = {}

# plt.figure()
# plt.imshow(images[0], cmap='gray')
# plt.title("Input image")
# plt.show()

# ネットワーク作成
n_inp = 784
n_e = 100
n_i = 100

N.append(neuron_inp(n_inp, max_rate=100))
N.append(neuron_e(n_e, "N_middle"))
N.append(neuron_i(n_i, "N_output"))

S.append(synapse_stdp(N[0], N[1], "S_0", connect=True)) # 入力層から興奮ニューロン
S.append(synapse_e(N[1], N[2], "exc", "S_1", delay=0*ms, connect="i==j")) # 興奮ニューロンから抑制ニューロン
S.append(synapse_i(N[2], N[1], "inh", "S_2", delay=0*ms, connect="i!=j")) # 側抑制
    
for i, neuron in enumerate(N):
    spikemon[i] = SpikeMonitor(neuron)
    
    if i != 0:
        statemon_n[i] = StateMonitor(neuron, ["v", "I_noise", "Ie", "Ii", "ge", "gi"], record=True)
        
for i, synapse in enumerate(S):
    statemon_s[i] = StateMonitor(synapse, ["w"], record=0)

# シミュレーションの実行 ===============================================================
print("[RUNNING SIMULATION...]")
network = Network(N, S, spikemon, statemon_n, statemon_s)
plotter.weight_plot(S[0], n_pre=n_inp, n_post=n_e, title="Initial weight plot of S0")
for i in range(n_samples):
    neuron_inp.change_image(images[i])
    network.run(350*ms)
    
    neuron_inp.change_image(np.zeros((28, 28)))
    network.run(150*ms)
# =====================================================================================

# シミュレーションの結果のプロット
print("[PLOTTING RESULTS...]")
for i in range(3):
    plt.figure()
    plotter.raster_plot(spikemon[i], 1, 1, network.t, f"Raster plot of N{i}")
    
plotter.weight_plot(S[0], n_pre=n_inp, n_post=n_e, title="Initial weight plot of S0")

plt.figure()
plotter.state_plot(statemon_n[1], 0, "v", 6, 1, network.t, fig_title="State plot of N1")
plotter.state_plot(statemon_n[1], 0, "Ie", 6, 2, network.t)
plotter.state_plot(statemon_n[1], 0, "Ii", 6, 3, network.t)
plotter.state_plot(statemon_n[1], 0, "ge", 6, 4, network.t)
plotter.state_plot(statemon_n[1], 0, "gi", 6, 5, network.t)
plotter.state_plot(statemon_s[0], 0, "w", 6, 6, network.t)
plt.show()