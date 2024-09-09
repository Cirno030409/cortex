import json
from typing import Any

import numpy as np
from brian2 import NeuronGroup, PoissonGroup
from brian2.units import *
import pprint as pp


class Conductance_Izhikevich2003:
    
    def __init__(self, neuron_type:str):
        
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
        
        with open("Izhikevich2003_parameters.json", "r") as file:
            self.all_params = json.load(file)
        
        try:
            self.params.update(self.all_params[neuron_type])
        except KeyError:
            raise ValueError(f"Neuron type {neuron_type} not found in Izhikevich2003_parameters.json")
                
    def __call__(self, N, exc_or_inh:str, *args, **kwargs) -> Any:
        if exc_or_inh == "exc":
            self.params.update({
                "refractory" : 0 * ms,
                "v_th" : -50
            })
        elif exc_or_inh == "inh":
            self.params.update({
                "refractory" : 2 * ms,
                "v_th" : -40
            })
        else :
            raise Exception("Neuron type must be 'exc' or 'inh'")
        neuron =  NeuronGroup(N, model=self.model, threshold="v>(v_th + theta)", reset="v=c; u+=d; theta+=theta_dt", refractory="refractory", method="euler", namespace=self.params, *args, **kwargs)
        neuron.v = self.params["v_reset"]
        neuron.ge = 0
        neuron.gi = 0
        return neuron
    
class Conductance_LIF:
    
    def __init__(self, params=None):
        if params is None:
            # パラメータ未指定時のデフォルトのパラメータ
            print("[WARNING] No parameters were specified for STDP synapse. Using default parameters as below.")
            self.params = {
                "I_noise"       : 0,        # 定常入力電流
                "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
                "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
                "taum"          : 10*ms,    # 膜電位の時定数
                "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
                "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
                "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
                "theta_dt"      : 0,        # ホメオスタシスの発火閾値の上昇値
                "refractory"    : 2 * ms,   # 不応期
                "v_reset"       : -60,      # リセット電位
                "v_rest"        : -50,      # 静止膜電位
                "v_th"          : -40       # 発火閾値
            }
            pp.pprint(self.params)
        else:
            self.params = params
        self.model = """
            dv/dt = ((v_rest - v) + (Ie + Ii + I_noise)) / taum : 1 (unless refractory)
            dge/dt = (-ge)/tauge : 1
            dgi/dt = (-gi)/taugi : 1
            dtheta/dt = -theta/tautheta : 1
            Ie = ge * (v_rev_e - v) : 1
            Ii = gi * (v_rev_i - v) : 1
        """      

        
    def __call__(self, N, exc_or_inh:str=None, *args, **kwargs) -> Any:
        # if exc_or_inh == "exc":
        #     self.params.update({
        #         "refractory" : 2 * ms,
        #         "v_reset" : -60,
        #         "v_th" : -40
        #     })
        # elif exc_or_inh == "inh":
        #     self.params.update({
        #         "refractory" : 2 * ms,
        #         "v_reset" : -60,
        #         "v_th" : -40
        #     })
        # else :
        #     raise Exception("Neuron type must be 'exc' or 'inh'")
        neuron =  NeuronGroup(N, model=self.model, threshold="v>(v_th + theta)", reset="v=v_reset; theta+=theta_dt", refractory="refractory", method="euler", namespace=self.params, *args, **kwargs)
        neuron.v = self.params["v_reset"]
        neuron.ge = 0
        neuron.gi = 0
        return neuron
        
    def change_params(self, neuron, params):
        """
        ニューロンのパラメータを変更する。

        Args:
            neuron (brian2.NeuronGroup): ニューロンのインスタンス
            params (dict): ニューロンのパラメータ
            ex:
                neuron.set_states({"v_reset": -50})
        """
        neuron.set_states(params)
        print("Neuron parameters were changed:")
        for key, value in params.items():
            print(f"{key}: {value}")
        
    def input_voltage(self, neuron, voltage):
        neuron.v_rev_e = voltage
        
class Poisson_Input:
        
    def __call__(self, N, max_rate:float, *args, **kwargs):
        self.max_rate = max_rate
        self.rates = np.zeros(N)

        self.neuron = PoissonGroup(N, self.rates * Hz, *args, **kwargs)
        return self.neuron
    
    def change_image(self, image:np.array, spontaneous_rate:int=0):
        self.rates = self.get_rate_from_image(image, spontaneous_rate)
        self.neuron.rates = self.rates
        
    def get_rate_from_image(self, image:list, spontaneous_rate):
        image = np.array(image).astype(float)
        if image.max() != 0:
            image /= image.max()  # 最大値がゼロの場合、正規化を行わない
            image *= self.max_rate
        # 0の部分をspontaneous_rateに置き換える
        image[image == 0] = spontaneous_rate
        # pp.pprint(image)
        # input()

        return image.flatten() * Hz
        