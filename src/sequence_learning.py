from seq2seq.models import Seq2seq
from keras.layers import Embedding
from keras.models import Sequential

import numpy as np
from itertools import zip_longest

def get_embedding_model(in_dim, out_dim = 300):
    '''
    Returns a model that generates transforms data from no. of input dimensions to output_dimensions
    @param in_dim: size of vocabulary
    @param out_dim: size of output representation for a term in vocabulary.
    @return model: embedding model for in_dim -> out_dim transformation.
    '''
    model = Sequential()
    model.add(Embedding(input_dim = in_dim, output_dim = out_dim, init='uniform', mask_zero= True, dropout=0.0))
    model.compile('rmsprop', 'mse')
    
    return model

def get_embeddings(data_in, model):
    '''
    Gets embeddings for data using an embedding model
    @param data_in: the input data for which we want to generate a dense representation
    @param model: embedding model for in_dim -> out_dim transformation.
    @return embedding of given data
    '''
    return model.predict(data_in)

def get_dense_data(data, model):
    '''
    Generates dense representation of given data using an embedding layer. 
    @param data: input data - 3 dimensional. 
                1st dim: instances
                2nd dim: timesteps (TS) for the given instance.
                3rd dim: representation of the TS, e.g. words/concepts in that TS.
    @param model: embedding model generated using get_embedding_model() for in_dim -> out_dim transformation.
    @return dense representation of data. Generates a dense representation for each term in vocab. 
            Averages the embeddings for all terms in the current TS to get a dense repr of the TS.
    '''
    new_data =list()
    for cur_data in data:
#         print(cur_data)
        cur_data = np.array(list(zip(*zip_longest(*cur_data, fillvalue=0))))
        cur_data_emb = np.average(get_embeddings(cur_data, model), axis = 1)
#         print(cur_data_emb)
        new_data.append(cur_data_emb.tolist())
        
    return new_data
        
def get_seq2seq_ae(data, n_hid = 300, n_layers_enc = 1, n_layers_dec = 1, broadcast_state=True, inner_broadcast_state=True, peek=False, corruption=0.05, batch_size = 128):
    '''
    Returns a seq2seq model based on parameters in data.
    @param data: the data_set to base model on. Used to calculate maxlen and input_dim (see further)
    @param maxlen: maximum number of time steps in input. Calculated from data
    @param input_dim: representation size of each time step
    @param n_hid: number of hidden nodes in each layer
    @param n_layers_enc: depth of encoder
    @param n_layers_dec: depth of decoder
    For an autoencoder setup, input is the same as output
    @return seq2seq autoencoder
    '''
    maxlen = len(data[0]) #input_length
    in_dim = len(data[0][0]) #input_dim

    model = Seq2seq(
        batch_input_shape = (batch_size, maxlen, in_dim),
        input_dim = in_dim, #w2v vector dimension. remove hard coding of value
        input_length = maxlen,
        hidden_dim = n_hid, #how many hidden nodes to use?
        output_dim = in_dim,
        output_length = maxlen,
        depth= (n_layers_enc, n_layers_dec),
        broadcast_state= broadcast_state, 
        inner_broadcast_state=inner_broadcast_state,
        peek=peek, 
        dropout=corruption
    )

#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='mse', optimizer='rmsprop')
    
    return model