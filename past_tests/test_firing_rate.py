"""
入力電流でどれくらいの発火率でニューロンが発火するかを確かめる。
"""
import MiniColumn as mc

import matplotlib.pyplot as plt
from brian2 import *

from MiniColumn import *

S = {}  # Synapsesの辞書
mc = {}  # MiniColumnのリスト

poisson_input = PoissonGroup(1, rates=0 * Hz)

mc[0] = MiniColumn(
    n_l4=1,
    n_l23=1,
    column_id=0,
    synapse_between_same_layer=False,
    neuron_model="Izhikevich2003",
)  # ミニカラムの定義

S["inp"] = Synapses(
    poisson_input, mc[0].N["l4"], "w : 1", on_pre="I += w", delay=1 * ms
)
S["inp"].connect("i==j") # inp -> l4
S["inp"].w = 1

net = Network(poisson_input, mc[0].network, S)

print("running simulation...")
net.run(1 * second)
print("done")

mc[0].draw_current()
mc[0].draw_potential()
# mc[0].draw_spike_trace(pre_synapse_num=[100, 110, 120], post_synapse_num=[100, 110, 120])
mc[0].draw_weight_changes(one_fig=True)
mc[0].draw_weight()
mc[0].draw_raster_plot()
mc[0].draw_conductance()
print(mc[0].get_firing_rate())

plt.show()
