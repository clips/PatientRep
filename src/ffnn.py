from keras.models import Model
from keras.layers import Input, Dense

def get_ffnn_model(n_in, n_out, n_hid_layers, n_hid, act_fn = 'relu', output_act = 'softmax'):
    """
    Get a feedforward neural network classification model with the given number of hidden layers
    @param n_in: number of input nodes
    @param n_out: number of output nodes
    @param n_hid_layers: number of hidden layers
    @param n_hid: list with number of hidden units for each hidden layer
    @param act_fn: activation function
    @return model 
    """
    layer_in = Input(shape = (n_in,))
    
    layer_final = layer_in
    
    if n_hid_layers > 1 and len(n_hid) == 1:
        n_hid = n_hid * n_hid_layers
        
    for cur_layer in range(n_hid_layers):
        cur_hid = Dense(n_hid[cur_layer], activation=act_fn, init = 'glorot_uniform')(layer_in)
        layer_final = cur_hid
        
    layer_out = Dense(n_out, activation=output_act, init = 'glorot_uniform')(layer_final)
    
    model = Model(layer_in, layer_out)
    
    return model