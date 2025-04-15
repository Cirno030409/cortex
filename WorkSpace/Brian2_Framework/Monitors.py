"""
自作のモニターのクラスを定義します。
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
    各記録変数の単位情報も保存します。
    """
    def __init__(self, monitor, record_variables):
        self.name = monitor.name
        
        # 時間データの処理と保存
        if hasattr(monitor.t, 'dimensions'):
            # 単位を取得
            unit = monitor.t.get_best_unit()
            if str(unit) == 's':  # 秒単位の場合はミリ秒に変換
                self.t = np.array(monitor.t / second * 1000)  # 秒からミリ秒へ変換
                self.t_unit = 'ms'
            else:
                self.t = np.array(monitor.t)
                self.t_unit = str(unit)
        else:
            # すでにnumpy配列の場合
            self.t = np.array(monitor.t)
            self.t_unit = 'ms'  # デフォルトはミリ秒とする
            
        self.source = [0] * len(monitor.source)
        self.N = len(monitor.source)  # ニューロン数を保存
        self.record_variables = record_variables  # record_variables属性を明示的に保存
        
        # 単位情報を保存する辞書
        self.units = {}
        
        # 各記録変数のデータを属性として保存
        for var in record_variables:
            try:
                # getattr(monitor, var)でmonitor.v, monitor.ge などのデータを取得
                # 全ニューロンの情報を保存 (N x T の行列)
                data = np.array(getattr(monitor, var))
                if len(data.shape) == 1:  # 1次元の場合は2次元に拡張
                    data = data.reshape(1, -1)
                self.__dict__[var] = data  # データを属性として保存
                
                # 単位情報を保存
                if hasattr(getattr(monitor, var), 'dimensions'):
                    self.units[var] = str(getattr(monitor, var).get_best_unit())
                else:
                    self.units[var] = '1'  # 単位なしの場合は1とする
            except AttributeError:
                print(f"[WARNING] {var} は {monitor.name} に存在しません。このパラメータは保存されません。")
                
    def extend(self, monitor):
        """
        モニターのデータを結合します。
        
        Args:
            monitor (StateMonitorData): 結合するモニター
        """
        self.t = np.concatenate([self.t, monitor.t])
        
        # 結合するモニターの単位情報を継承
        if hasattr(monitor, 'units'):
            for var, unit in monitor.units.items():
                if var in self.record_variables:
                    # 単位が異なる場合は警告を表示
                    if var in self.units and self.units[var] != unit:
                        print(f"[WARNING] 変数 {var} の単位が異なります。{self.units[var]} と {unit}")
                    # 単位情報を更新または追加
                    self.units[var] = unit
        
        # データを結合
        for var in self.record_variables:
            if hasattr(monitor, var):
                self.__dict__[var] = np.concatenate([self.__dict__[var], getattr(monitor, var)])
            
    def get_best_unit(self, var=None):
        """
        変数の最適な単位を返します。
        
        Args:
            var (str): 単位を取得する変数名。Noneの場合は時間の単位を返します。
            
        Returns:
            str: 単位を表す文字列
        """
        if var is None:
            return self.t_unit
        elif var in self.units:
            return self.units[var]
        else:
            return '1'  # 単位なしの場合は1を返す
