import random


def doc2vec_parameter_space():
    return {
        "dimension": list(range(50, 1550, 50)),
        "win_size": list(range(1, 11)),
        "min_freq": list(range(1, 11)),
        "neg_samples": list(range(1, 11)),
        "iter": list(range(5,21)),
        "dm": [0, 1], #0: dbow, 1: dm
        
    }

def sdae_parameter_space():
    return {
        "n_hid": list(range(50, 1550, 50)),
        "n_hid_layers": list(range(1, 6)),
        "dropout": [float(x)/100 for x in range(0,90,5)]
        
    }
    
def ffnn_parameter_space():
    return {
            "n_hid_layers" : list(range(0,11)),
            "n_hid": list(range(10, 1010, 10)),
            "activation": ['sigmoid', 'tanh', 'relu']  
            }

class RandomizedSearch():
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space()

    def size_parameter_space(self):
        size = 1
        for param, val in self.parameter_space.items():
            size *= len(val)
        return size

    def sample(self):
        return {param: self.draw(val) for param, val in self.parameter_space.items()}

    def draw(self, lst):
        return random.choice(lst)
