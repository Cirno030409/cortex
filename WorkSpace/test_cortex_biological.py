from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *

seed = 0

save_dir = os.path.join("data", "cortex_biological", "single", "without noize", f"seed={seed}")
os.makedirs(save_dir, exist_ok=True)

np.random.seed(seed)
defaultclock.dt = 0.5*ms
duration = 100*ms

params = load_parameters("Brian2_Framework/parameters/Cortex_Microcircuit/params.json")

net = Jung_H_Lee_Cortex_MicroCircuit(params)
visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))

# net.run(duration, report="text")

# plot_all_monitors(net.network, save_dir_path=save_dir)
# save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))

show()
