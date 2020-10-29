import numpy as np
import theano
import keras.backend as K
from scipy.sparse import issparse

def get_sdae_input_significance(classifier_model, sdae_model, feats_classifier, feats_sdae):
    """
    Calculates the significance of original input features for a task across a 2 level architecture.
    In the first level, a denoising autoencoder is used to generate dense patient representation.
    In the second level, these dense representation are used as input features for a classification task.
    The model performs backpropagation across the two networks to calculate input significance.
    @param classifier_model: trained model for feed forward neural network for classification task
    @parma sdae_model: trained model for denoising autoencoder that generates dense represenation
    @param feats_classifier: input features for classifier
    @param feats_sdae: input features for denoising autoencoder
    @return significance of input features for a task, across all instances
    @return significance of input unit for every output unit. Shape: (n_input, n_output)
    """
    sensitivity_classifier_output_hid_inst = _get_output_sensitivity(classifier_model, feats_classifier)
    sensitivity_sdae_hid_input_inst = _get_hidden_sensitivity(sdae_model, feats_sdae)
    
    sensitivity_output_input_inst = _get_output_input_sensitivity(sensitivity_classifier_output_hid_inst, sensitivity_sdae_hid_input_inst)
    
    significance_input_output = _get_significance_input_output(sensitivity_output_input_inst)
    input_significance = _get_input_significance(significance_input_output)
    
    return input_significance, significance_input_output
    
def get_ffnn_input_significance(model, feats):
    """
    Calculate significance of input features for a feed forward neural network
    @param model: keras feed forward neural network model
    @param feats: input feature set
    @return input_significance: 1D array with significance score of each feature
    @return significance of input unit for every output unit. Shape: (n_input, n_output)
    """
    sensitivity_output_input_inst = _get_output_sensitivity(model, feats)
    significance_input_output = _get_significance_input_output(sensitivity_output_input_inst)
    input_significance = _get_input_significance(significance_input_output)
    
    return input_significance, significance_input_output

def _get_input_significance(significance_input_output):
    """
    Returns significance of every input unit, which is equal to the maximum significance it generates among all output units
    @param significance_input_output: Significance of every input unit for every output unit
                                      Shape: (n_input, n_output)
    @return 1D array with significance score of each feature
    """
    return  np.amax(significance_input_output, axis = 1)


def _get_significance_input_output(sensitivity_output_input_inst):
    """
    Calculates significance of an input unit with respect to an output unit across all instances.
    This value is the root mean square of sensitivity of an output unit with respect to the input unit across all instances.
    @param sensitivity_output_input_inst: For every instance, sensitivity of every output node with respect to every input node.
                                          Shape (n_inst, n_output, n_input)
    @return significance_input_output: Significance of every input unit for every output unit
                                       Shape: (n_input, n_output)
    """
    significance_input_output = np.sqrt(np.mean(np.square(sensitivity_output_input_inst), axis = 0)).transpose()
    return significance_input_output

def _get_output_input_sensitivity(sensitivity_classifier_output_hid_inst, sensitivity_sdae_hid_input_inst):
    """
    Get sensitivity of output units with respect to input units, given sensitivity of output units wrt hidden units, and hidden units wrt input units.
    Chain rule across networks
    @param sensitivity_classifier_output_hid_inst: sensitivity of output units wrt input units for all instances for a classifier
    @param sensitivity_sdae_hid_input_inst: sensitvity of hidden units wrt input units for all instances for a classifier
    @return sensitivity of output units wrt input units for each instance
    """
    sensitivity_output_input_inst = np.zeros(shape=(sensitivity_classifier_output_hid_inst.shape[0], sensitivity_classifier_output_hid_inst.shape[1], sensitivity_sdae_hid_input_inst.shape[2]))
    for i in range(sensitivity_classifier_output_hid_inst.shape[0]):
        sensitivity_output_input_inst[i] = np.dot(sensitivity_classifier_output_hid_inst[i], sensitivity_sdae_hid_input_inst[i])
        
    return sensitivity_output_input_inst

def _get_output_sensitivity(model, feats):
    """
    For every instance, get sensitivity of every output node with respect to every input node
    Assumes theano backend
    @param model: keras model
    @param feats: input features
    @return sensitivity_output_input_inst: For every instance, sensitivity of every output node with respect to every input node.
                                           Shape (n_inst, n_output, n_input)
    """
    
    #if not issparse(feats) and len(feats.shape) == 1:
        

    n_inst = feats.shape[0]
    y = model.output
    x = model.input
    
    print("Number of input nodes:", model.input_shape)
    #feats_tensor = T.as_tensor_variable(feats)
    #J, updates = theano.scan(lambda i, y, x : theano.gradient.jacobian(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
    #J = theano.gradient.jacobian(y[0,0],x)
    #f = theano.function([x], [J, J.shape])#, updates=updates)
    #print(f(feats)[0])
    
    sensitivity_output_input_inst = np.zeros(shape=(n_inst, model.output_shape[1], model.input_shape[1]))
    for i in range(n_inst):
        print("Getting Jacobian for instance", i)
        
        if issparse(feats):
            feats = feats.getrow(i).todense()
        else:
            feats = feats[i]
            #feats = feats[np.newaxis, :]
        
        for j in range(model.output_shape[1]):
            J = theano.gradient.jacobian(y[i,j],x)
            f = theano.function([x], J) #[J, J.shape])
            #print(f(feats)[i])
            sensitivity_output_input_inst[i,j] = f(feats)[0] # index 0 because we are calculating output of cur_feat, which is only one instance at a time. Otherwise make it i
    
    
    return sensitivity_output_input_inst

def _get_hidden_sensitivity(model, feats):
    """
    For every instance, get sensitivity of every hidden node in the last hidden layer with respect to every input node
    Assumes theano backend
    @param model: keras model
    @param feats: input features
    @return sensitivity_output_hid_inst: For every instance, sensitivity of every hidden node from last layer with respect to every input node.
                                           Shape (n_inst, n_output, n_input)
    """    
    n_inst = feats.shape[0]
    
    y = model.layers[-1]
    x = model.input
    
    train = 0
    
    print("Number of hidden nodes:", y.output_shape)
    
    sensitivity_output_hid_inst = np.zeros(shape=(n_inst, y.output_shape[1], model.input_shape[1]))
    for i in range(n_inst):
        print("Getting Jacobian for instance", i)
        
        if issparse(feats):
            cur_feats = feats.getrow(i).todense()
        else:
            cur_feats = feats[i]
        
        for j in range(y.output_shape[1]):
            J = theano.gradient.jacobian(y.output[0,j],x) # index 0 because we are calculating output of cur_feat, which is only one instance at a time. Otherwise make it i
            f = K.function([x, K.learning_phase()], J)
            sensitivity_output_hid_inst[i,j] = f([cur_feats,train])[0] # index 0 because we are calculating output of cur_feat, which is only one instance at a time. Otherwise make it i
            
    return sensitivity_output_hid_inst
