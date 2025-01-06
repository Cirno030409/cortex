import json
from typing import Any

import numpy as np
from brian2 import NeuronGroup, PoissonGroup
from brian2.units import *
import pprint as pp

class Conductance_LIF_Neuron(NeuronGroup):
    
    def __init__(self, N, params=None, *args, **kwargs):
        if params is None:
            raise ValueError("ニューロンを作成する際、パラメータを必ず指定する必要があります。")
        self.params = params
        self.model = """
            dv/dt = ((v_rest - v) + (Ie + Ii + I_noise)) / taum : 1 (unless refractory)
            dge/dt = (-ge)/tauge : 1
            dgi/dt = (-gi)/taugi : 1
            dtheta/dt = -theta/tautheta : 1
            Ie = ge * (v_rev_e - v) : 1
            Ii = gi * (v_rev_i - v) : 1
        """
        super().__init__(N, model=self.model, threshold="v>(v_th + theta)", 
                         reset="v=v_reset; theta+=theta_dt", refractory="refractory", 
                         method="euler", namespace=self.params, *args, **kwargs)
        self.v = self.params["v_reset"]
        self.ge = 0
        self.gi = 0

    def change_params(self, params):
        """
        ニューロンのパラメータを変更する。

        Args:
            params (dict): ニューロンのパラメータ
        """
        for key, value in params.items():
            self.params[key] = value  # パラメータ辞書を更新
        
        print("Neuron parameters were changed:")
        for key, value in params.items():
            print(f"{key}: {value}")

class Conductance_Izhikevich2003(NeuronGroup):
    # NOTE 動作未確認
    def __init__(self, N, neuron_type:str, *args, **kwargs):
        self.model = """
            dv/dt = (0.04*v**2 + 5*v + 140 - u + Ie + Ii + I_noise)/ms : 1 (unless refractory)
            du/dt = (a*(b*v - u))/ms : 1
            dge/dt = (-ge)/tauge : 1
            dgi/dt = (-gi)/taugi : 1
            dtheta/dt = -theta/tautheta : 1
            Ie = ge * (v_rev_e - v) : 1
            Ii = gi * (v_rev_i - v) : 1
        """
        
        self.params = {
            "I_noise"       : 0,
            "tauge"         : 2*ms,
            "taugi"         : 30*ms,
            "tautheta"      : 1e7*ms,
            "v_reset"       : -65,
            "v_rev_e"       : 0,
            "v_rev_i"       : -100,
            "theta_dt"      : 0,
        }
        
        with open("parameters/Izhikevich2003_parameters.json", "r") as file:
            self.all_params = json.load(file)
        
        try:
            self.params.update(self.all_params[neuron_type])
        except KeyError:
            raise ValueError(f"Neuron type {neuron_type} not found in Izhikevich2003_parameters.json")
        
        super().__init__(N, model=self.model, threshold="v>(v_th + theta)", reset="v=c; u+=d; theta+=theta_dt", refractory="refractory", method="euler", namespace=self.params, *args, **kwargs)
        self.v = self.params["v_reset"]
        self.ge = 0
        self.gi = 0
    
        
class Poisson_Input_Neuron(PoissonGroup):
    def __init__(self, N, max_rate:float, *args, **kwargs):
        self.max_rate = max_rate
        self.rates = np.zeros(N)
        super().__init__(N, self.rates * Hz, *args, **kwargs)
    
    def change_image(self, image:np.array, spontaneous_rate:int=0):
        # self.rates = self.get_rate_from_image(image, spontaneous_rate)
        self.rates = self.get_rate_from_image_equalize_sum(image, 4480)
        self.rates_ = self.rates
        
    def get_rate_from_image_equalize_sum(self, image:np.array, goal_sum:int):
        """
        画素の合計値がgoal_sumになるように正規化します。

        Args:
            image (np.array): 正規化する画像
            goal_sum (int): 正規化後の画素の合計値
        """
        # 画像の画素値の合計を計算
        current_sum = np.sum(image)
        
        # 0除算を避けるためのチェック
        if current_sum == 0:
            return np.zeros_like(image)
        
        # 正規化係数を計算
        normalization_factor = goal_sum / current_sum
        
        # 画像を正規化
        normalized_image = image * normalization_factor
        
        return normalized_image.flatten() * Hz

    def get_rate_from_image(self, image:list, spontaneous_rate):
        image = np.array(image).astype(float)
        if image.max() != 0:
            image /= image.max()  # 最大値がゼロの場合、正規化を行わない
            image *= self.max_rate
        # 0の部分をspontaneous_rateに置き換える
        image[image == 0] = spontaneous_rate

        return image.flatten() * Hz
        