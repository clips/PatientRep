'''
@author madhumita
'''
import numpy as np
import scipy.sparse as scp

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.utils.visualize_util import plot

def train(model, x_train, y_train, x_val, y_val, n_classes, loss = 'categorical_crossentropy', optimizer = 'rmsprop', batch_size = 32, nb_epoch = 20, verbose = 1):
    """
    Train a neural network classification model based on given parameter settings
    @param x_train: training feats
    @param y_train: training labels
    @param x_val: validation feats
    @param y_val: validation labels
    @param n_classes: number of output classes (to convert labels to one hot representation)
    @param loss: loss during training process
    @param optimizer: optimizer to use during training
    @param batch_size: mini-batch size during model training
    @param nb_epoch: number of epochs to train the model for
    @param verbose: 1 to print logs verbose
    @return trained model
    """    
    model.compile(loss = loss, optimizer=optimizer, metrics=['accuracy'])
        
    #Early stopping to stop training when val loss increses for 1 epoch
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0)
        
    model.fit_generator(generator = batch_generator(x_train, y_train,
                                                    batch_size = batch_size, 
                                                    shuffle = True, 
                                                    n_classes=n_classes,
                                                    one_hot_labels= True),
                        samples_per_epoch = x_train.shape[0], 
                        callbacks = [early_stopping], 
                        nb_epoch=nb_epoch,
                        verbose=verbose, 
                        validation_data  = batch_generator(x_val, y_val, 
                                                           batch_size = batch_size, 
                                                           shuffle = False, 
                                                           n_classes=n_classes,
                                                           one_hot_labels= True), 
                        nb_val_samples = x_val.shape[0],
                        nb_worker = 1
                        )
    
    return model

def evaluate_on_test(fit_model, x_test, y_test, n_classes, cfg, batch_size = 32):
    """
    Evaluate a trained model on test dataset 
    Use this function only for the final evaluation, not for development
    """
    fit_model.evaluate_generator(generator = batch_generator(x_test, y_test,
                                                             batch_size = batch_size, 
                                                             shuffle = False,
                                                             one_hot_labels= True,
                                                             n_classes = n_classes),
                                samples = x_test.shape[0],
                                )

def predict(fit_model, x_test, cfg, batch_size = 32):
    """
    Get prediction probability for each class for test data
    @param fit_model: trained model
    @param x_test: test data
    @patam cfg: config object
    """
    predictions = fit_model.predict_generator(generator = batch_generator(x_test, None,
                                                                          batch_size = batch_size, 
                                                                          shuffle = False, 
                                                                          y_gen = False
                                                                          ),
                                              val_samples = x_test.shape[0],
                                              )
    return predictions

def save_model(model, out_dir, f_arch = 'model_arch.png', f_model = 'model_arch.json', f_weights = 'model_weights.h5'):
    '''
    Saves a Keras model description and model weights
    @param model: a keras model
    @param out_dir: directory to save model architecture and weights to
    @param f_model: file name for model architecture
    @param f_weights: filename for model weights
    '''
    model.summary()
    plot(model, to_file=out_dir+f_arch)
    
    json_string = model.to_json()
    open(out_dir+f_model, 'w').write(json_string)
    
    model.save_weights(out_dir+f_weights, overwrite=True)

def load_model(dir_name, f_model = 'model_arch.json', f_weights = 'model_weights.h5' ):
    '''
    Loads a Keras model from disk to memory.
    @param dir_name: directory in which the model architecture and weight files are present
    @param f_model: file name for model architecture
    @param f_weights: filename for model weights
    @return loaded model
    '''
    json_string = open(dir_name + f_model, 'r').read()
    model = model_from_json(json_string)
    
    model.load_weights(dir_name+f_weights)
    
    return model


def batch_generator(X, Y, batch_size, shuffle, y_gen = True, n_classes = None, one_hot_labels = False):
    '''
    Creates batches of data from given dataset, given a batch size. Returns dense representation of x_sparse input.
    @param X: input features, x_sparse or dense
    @param Y: input labels, x_sparse or dense
    @param batch_size: number of instances in each batch
    @param shuffle: If True, shuffle input instances.
    @return batch of input features and labels
    '''
    number_of_batches = np.ceil(X.shape[0]/batch_size) #ceil function allows for creating last batch off remaining samples
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    
    x_sparse = False
    y_sparse = False
    if scp.issparse(X):
        x_sparse = True
    if y_gen and scp.issparse(Y):
        y_sparse = True
    
    
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if x_sparse:
            x_batch = X[batch_index,:].toarray() #converts to dense array    
        else:
            x_batch = X[batch_index,:]
        
        if y_gen:
            if y_sparse:
                y_batch = Y[batch_index,:].toarray() #converts to dense array
            elif type(Y) is list:
                y_batch = [Y[i] for i in batch_index]
            else:
                y_batch = Y[batch_index,:]
        
            if one_hot_labels:
                y_batch = to_categorical(y_batch, n_classes)
            yield x_batch, y_batch
            
        else:
            yield x_batch
            
        counter += 1
        
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def x_generator(X, batch_size, shuffle, cont_gen = False):
    '''
    Creates batches of data from given input, given a batch size. Returns dense representation of sparse input one batch a time.
    @param X: input features, can be sparse or dense
    @param batch_size: number of instances in each batch
    @param shuffle: If True, shuffle input instances.
    @param cont_gen: If True, generate batches continuously, else generate for fixed number of batches
    @return batch of input data, without shuffling
    '''
    number_of_batches = np.ceil(X.shape[0]/batch_size) #ceil function allows for creating last batch off remaining samples
    
    sample_index = np.arange(X.shape[0])
    
    if shuffle:
        np.random.shuffle(sample_index)
    
    sparse = False
    if scp.issparse(X):
        sparse = True
    
    counter = 0
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if sparse:
            x_batch = X[batch_index,:].toarray() #converts to dense array
        else:
            x_batch = X[batch_index,:]
        yield x_batch, batch_index
        counter += 1
        if not cont_gen and counter >= number_of_batches:
            break
        elif cont_gen and (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
