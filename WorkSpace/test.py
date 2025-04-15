from brian2 import *
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100) * ms
y = np.linspace(0, 10, 100) * nA

plt.plot(x, y)
plt.show()
