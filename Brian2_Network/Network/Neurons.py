import json
from typing import Any

from brian2 import NeuronGroup
from brian2.units import *


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
            "I_noise"       : 20,
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
                
    def __call__(self, N, exc_or_inh:str, name:str, *args, **kwargs) -> Any:
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
        neuron =  NeuronGroup(N, model=self.model, threshold="v>(v_th + theta)", reset="v=c; u+=d; theta+=theta_dt", refractory="refractory", method="euler", namespace=self.params, name=name, *args, **kwargs)
        neuron.v = self.params["v_reset"]
        neuron.ge = 0
        neuron.gi = 0
        return neuron
    
class Conductance_LIF:
    
    def __init__(self):
        
        self.model = """
            dv/dt = ((v_reset - v) + (Ie + Ii + I_noise)) / taum : 1 (unless refractory)
            dge/dt = (-ge)/tauge : 1
            dgi/dt = (-gi)/taugi : 1
            dtheta/dt = -theta/tautheta : 1
            Ie = ge * (v_rev_e - v) : 1
            Ii = gi * (v_rev_i - v) : 1
        """
        
        self.params = {
            "I_noise"       : 20,
            "tauge"         : 2*ms,
            "taugi"         : 30*ms,
            "taum"          : 10*ms,
            "tautheta"      : 1e7*ms,
            "v_rev_e"       : 0,
            "v_rev_i"       : -100,
            "theta_dt"      : 0,
        }      

        
    def __call__(self, N, exc_or_inh:str, name:str, *args, **kwargs) -> Any:
        if exc_or_inh == "exc":
            self.params.update({
                "refractory" : 0 * ms,
                "v_reset" : -65,
                "v_th" : -50
            })
        elif exc_or_inh == "inh":
            self.params.update({
                "refractory" : 2 * ms,
                "v_reset" : -45,
                "v_th" : -40
            })
        else :
            raise Exception("Neuron type must be 'exc' or 'inh'")
        neuron =  NeuronGroup(N, model=self.model, threshold="v>(v_th + theta)", reset="v=v_reset; theta+=theta_dt", refractory="refractory", method="euler", namespace=self.params, name=name, *args, **kwargs)
        neuron.v = self.params["v_reset"]
        neuron.ge = 0
        neuron.gi = 0
        return neuron