from brian2 import *

def normalize_weight(synapses, a=0, b=1):
    """
    重みを0-1の範囲に正規化する

    Args:
        synapses (Synapses): Synapsesオブジェクトのリスト

    Returns:
        None
    """
    for synapse in synapses:
        min_w = np.min(synapse.w)
        print("min_w: ", min_w)
        max_w = np.max(synapse.w)
        print("max_w: ", max_w)
        normalized_weights = (b - a) * (synapse.w - min_w) / (max_w - min_w) + a
        synapse.w = normalized_weights
