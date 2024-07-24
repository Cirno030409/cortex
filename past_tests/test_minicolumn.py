import matplotlib.pyplot as plt
from brian2 import *

from MiniColumn import *

S = {}  # Synapsesの辞書
mc = {}  # MiniColumnのリスト

poisson_input = PoissonGroup(5, rates=100 * Hz)
mc[0] = MiniColumn(
    n_l4=5,
    n_l23=5,
    column_id=0,
    initial_weight=np.linspace(0, 1, 5 * 5),
    synapse_between_same_layer=False,
    neuron_model="Izhikevich2003",
)  # ミニカラムの定義

S["inp"] = Synapses(
    poisson_input, mc[0].N["l4"], "w : 1", on_pre="I += w", delay=1 * ms
)
S["inp"].connect(i=[0], j=[0,1]) # inp -> l4
S["inp"].w = 1

net = Network(poisson_input, mc[0].network, S)

print("running simulation...")
net.run(1 * second)
print("done")

# mc[0].draw_current()
# mc[0].draw_potential()
mc[0].draw_spike_trace()
# mc[0].draw_weight(one_fig=True)
# mc[0].draw_raster_plot()

mc[0].get_firing_rate_per_neuron()

plt.show()
