import matplotlib.pyplot as plt
from brian2 import *

from MiniColumn import *

S = {}  # Synapsesの辞書
mc = {}  # MiniColumnのリスト

poisson_input = PoissonGroup(100, rates=100 * Hz)
# noize_input = PoissonGroup(100, rates=30 * Hz)
mc[0] = MiniColumn(
    n_l4=25,
    n_l23=9,
    column_id=0,
    synapse_between_same_layer=False,
    neuron_model="Izhikevich2003",
)  # ミニカラムの定義

#! LGN -> L4 線分を呈示
S["inp->l4"] = Synapses(
    poisson_input, mc[0].N["l4"], 
    model="""
    w : 1
    """, 
    on_pre="I_post += w", 
    delay=1 * ms
)
S["inp->l4"].connect(i=0, j=[11, 12, 13]) 
S["inp->l4"].w = 1

# #! ノイズ入力からL4へ(自発発火)
# S["noize->l4"] = Synapses(
#     noize_input, mc[0].N["l4"], "w : 1", on_pre="I += w", delay=1 * ms 
# )
# S["noize->l4"].connect("i == j")
# S["noize->l4"].w = 1

net = Network(poisson_input, mc[0].network, S)

print("running simulation...")
net.run(100 * second)
print("simulation has done")
print("plotting...")

# mc[0].draw_current()
mc[0].draw_potential()
# mc[0].draw_spike_trace(pre_synapse_num=[100, 110, 120], post_synapse_num=[100, 110, 120])
mc[0].draw_weight_changes(one_fig=True)
mc[0].draw_weight()
mc[0].draw_raster_plot()
print(mc[0].get_firing_rate())


print("done")

plt.show()
