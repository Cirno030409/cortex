import numpy as np
from brian2 import *

# 例: 5x5の画素値データ
pixel_values = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

# 画素値を0-1の範囲に正規化
normalized_pixel_values = pixel_values / 255.0

# 発火率を計算（最大発火率を100 Hzとする）
max_rate = 100 * Hz
firing_rates = normalized_pixel_values * max_rate

# PoissonGroupの作成
n_neurons = pixel_values.size
P = PoissonGroup(n_neurons, rates=firing_rates.flatten())

# モニターの設定
spikemon = SpikeMonitor(P)

# シミュレーションの実行
duration = 1000 * ms
run(duration)

# 結果のプロット
figure(figsize=(12, 4))
plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
title('Raster plot of spikes')
show()