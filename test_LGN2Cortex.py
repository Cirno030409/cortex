import matplotlib.pyplot as plt
from brian2 import *

from MiniColumn_fix import *

mc = {}  # MiniColumnのリスト

# 5x5の画素値データ
pixel_values = []
pixel_values.append(np.array([
    255,   0,   0,   0,   0,
    0,   0,   0,   0,   0,
    0, 255, 255, 255,   0,
    0,   0,   0,   0,   0,
    0,   0,   0,   0,   0
]))
# pixel_values.append(np.array([
#     0,   0,   0,   0,   0,
#     0,   0, 255,   0,   0,
#     0,   0, 255,   0,   0,
#     0,   0, 255,   0,   0,
#     0,   0,   0,   0,   0
# ]))
pixel_values.append(np.array([
    0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,
    0,   0,   0,   0,   0
]))

simulation_duration = 800 * ms

max_rate = 20 * Hz # 最大発火率
# 画素値を0-1の範囲に正規化
for i in range(len(pixel_values)):
    pixel_values[i] = pixel_values[i] / 255.0
    pixel_values[i] = pixel_values[i] * max_rate

input_data = TimedArray(pixel_values, dt=400 * ms)
input_neurons = PoissonGroup(25, rates="input_data(t, i)")

mc[0] = MiniColumn(
    simulation_duration=simulation_duration,
    n_l4=25,
    n_l23=9,
    n_inhibitory=9,
    column_id=0,
    input_neurons=input_neurons,
    synapse_between_same_layer=False,
    neuron_model="Izhikevich2003",
)  # ミニカラムの作成
net = Network(mc[0].network)

#! シミュレーションの実行 ================================================
print("[RUNNING SIMULATION...]")

net.run(simulation_duration)

print("[SIMULATION HAS DONE]")
print("[PLOTTING...]")
#! =====================================================================


mc[0].draw_current()
mc[0].draw_potential(neuron_num_l4=[11, 12, 13])
# mc[0].draw_spike_trace(pre_synapse_num=[100, 110, 120], post_synapse_num=[100, 110, 120])
# mc[0].draw_weight_changes(one_fig=True)
# mc[0].draw_weight()
mc[0].draw_raster_plot()
# mc[0].draw_conductance()
print(mc[0].show_firing_rate(50 * second))


print("[PLOTTING DONE]")

plt.show()
