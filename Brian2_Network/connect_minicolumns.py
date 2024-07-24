import numpy as np
from brian2.units import *
from Network.Networks import MiniColumn
from Network.Networks import Cortex


cortex = Cortex()
cortex.add_minicolumns(4, 1, 1, 1, 0, True)

cortex.connect_minicolumns(0, 1, condition="i==j")

cortex.run(np.ones(1), max_rate=70, duration=500 * ms)