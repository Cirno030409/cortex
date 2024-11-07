"""
モニターのクラスを定義します。
"""

import numpy as np
from brian2 import *
import pickle as pkl



class SpikeMonitorData():
    """
    SpikeMonitorのデータを保存するためのクラス
    pickleで保存できるようにするために実装。
    """
    def __init__(self, monitor):
        self.name = monitor.name
        self.i = np.array(monitor.i)  # numpy配列として保存
        self.t = np.array(monitor.t)
        self.count = np.array(monitor.count)
        self.source = [0] * len(monitor.source)
        # self.record = monitor.record
        # spike_trainsはweakrefを含むため、必要なデータのみ抽出
        self.spike_trains = {i: np.array(trains) for i, trains in monitor.spike_trains().items()}
        self.num_spikes = monitor.num_spikes
        
    def extend(self, monitor):
        """
        モニターのデータを結合します。

        Args:
            monitor (SpikeMonitor or StateMonitor): 結合するモニター
        """
        self.i = np.concatenate([self.i, monitor.i])
        self.t = np.concatenate([self.t, monitor.t])
        self.count = np.concatenate([self.count, monitor.count])
        self.spike_trains = {**self.spike_trains, **monitor.spike_trains}
        self.num_spikes = self.num_spikes + monitor.num_spikes

class StateMonitorData():
    """
    StateMonitorのデータを保存するためのクラス
    pickleで保存できるようにするために実装。
    """
    def __init__(self, monitor, record_variables):
        self.name = monitor.name
        self.t = np.array(monitor.t)
        self.source = [0] * len(monitor.source)
        self.N = len(monitor.source)  # ニューロン数を保存
        # self.record = record_variables
        # 各記録変数のデータを属性として保存
        for var in record_variables:
            try:
                # getattr(monitor, var)でmonitor.v, monitor.ge などのデータを取得
                # 全ニューロンの情報を保存 (N x T の行列)
                data = np.array(getattr(monitor, var))
                if len(data.shape) == 1:  # 1次元の場合は2次元に拡張
                    data = data.reshape(1, -1)
                self.__dict__[var] = data  # データを属性として保存
            except AttributeError:
                print(f"[WARNING] {var} は {monitor.name} に存在しません。このパラメータは保存されません。")
                
    def extend(self, monitor):
        """
        モニターのデータを結合します。
        """
        self.t = np.concatenate([self.t, monitor.t])
        for var in self.record_variables:
            self.__dict__[var] = np.concatenate([self.__dict__[var], getattr(monitor, var)])
