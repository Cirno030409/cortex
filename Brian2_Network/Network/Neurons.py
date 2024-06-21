import json
from typing import Any

from brian2 import NeuronGroup
from brian2.units import *


class Conductance_Izhikevich2003:
    
    def __init__(self, neuron_type:str):
        
        self.model = """
            dv/dt = (0.04*v**2 + 5*v + 140 - u + I + I_noise)/ms : 1 (unless refractory)
            du/dt = (a*(b*v - u))/ms : 1
            dgsyn/dt = (-gsyn)/taugsyn : 1
            I = gsyn * (v_rev - v) : 1
        """
        
        self.params = {
            "v_th"          : -50,
            "v_reset"       : -65,
            "refractory"    : 3*ms,
            "I_noise"       : 0,
            "taugsyn"       : 80*ms,
        }
        
        with open("Izhikevich2003_parameters.json", "r") as file:
            self.all_params = json.load(file)
        
        try:
            self.params.update(self.all_params[neuron_type])
        except KeyError:
            raise ValueError(f"Neuron type {neuron_type} not found in Izhikevich2003_parameters.json")
        
        

        
    def __call__(self, N, exc_or_inh:str, tag_name:str, *args, **kwargs) -> Any:
        if exc_or_inh == "exc":
            self.params.update({
                "v_rev" : 0
            })
        elif exc_or_inh == "inh":
            self.params.update({
                "v_rev" : -80
            })
        else :
            raise ValueError("Neuron type must be 'exc' or 'inh'")
        neuron =  NeuronGroup(N, model=self.model, threshold="v>v_th", reset="v=c; u+=d", refractory="refractory", name=tag_name, method="euler", namespace=self.params, *args, **kwargs)
        neuron.v = self.params["v_reset"]
        neuron.gsyn = 0
        return neuron