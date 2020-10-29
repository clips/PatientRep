# -*- coding: utf-8 -*-

# import MySQLdb #to use this import, first do a pip3 install mysqlclient

import pickle
import json
import gzip
import numpy as np
from collections import defaultdict, OrderedDict
import operator
from scipy.io import mmread, mmwrite

def connect_to_db(hname, port, uname, pwd, dbase):
    '''
    Connects to the MySQL database where data resides
    @return database connection
    '''
    db_conn = MySQLdb.connect(host = hname,
                           port = port,
                           user = uname,
                           passwd = pwd,
                           db = dbase)
    return db_conn

def write_txt_file(out_dir, fname, content):
    '''
    Write textual content to a file
    @param out_dir: the directory where file needs to be written
    @param fname: name of the out file
    @param content: the content to be written to the file
    '''
    with open(out_dir+fname, 'w') as f:
        f.write(content)

def write_labels(pred, fname):
    '''
    Writes the gold standard or predicted labels for a dataset in the MTX format.
    @param pred: scipy sparse matrix with instance row number and corresponding label IDs
    @param fname: output file name with path 
    '''
    mmwrite(fname, pred)

def read_labels(fname):
    '''
    Returs a sparse matrix of gold standard/predicted labels after reading it from the MTX format
    @param fname: file name (with path) with the MTX data.
    @return scipy sparse matrix with instance and correponding label IDs
    '''
    return mmread(fname)
    
def write_data(data, out_dir, f_name, pickle_data = False, compress = True):
    '''
    Writes given data to disc in zipped pickle format
    @param data: data to write
    @param out_dir: output directory
    @param fname: file name for data set on disc
    '''
    print("Saving dataset")
    if pickle_data:
        if compress:
            f = gzip.open(out_dir+f_name, 'wb') #w: write; b: binary
        else:
            f = open(out_dir+f_name, 'wb')
        pickle.dump(data, f,  protocol = pickle.HIGHEST_PROTOCOL) #protocol 2 is the newest protocol, allows for efficient pickling of new-style classes
    else:
        f = open(out_dir+f_name, 'w')
        json.dump(data, f)
    f.close()
    print("Dataset written to file")
    
def write_dataset_to_file(out_dir,f_name, x_train,y_train, x_val = None, y_val = None, x_test = None, y_test = None):
    '''
    Writes a dataset to a zipped pickled file
    @param x_train: training set feats
    @param y_train: training set labels
    @param x_val: validation set feats
    @param y_val: validation set labels
    @param x_test: test set feats
    @param y_test: test set labels
    @param out_dir: output directory
    @param f_name: file name for dataset on disc
    '''
    train_set = x_train,np.asarray(y_train)
    
    try:
        val_set = x_val,np.asarray(y_val)
    except:
        val_set = None
    
    try:    
        test_set = x_test,np.asarray(y_test)
    except:
        test_set = None
        
    dataset = [train_set, val_set, test_set]
    
    print("Saving dataset")
    with gzip.open(out_dir+f_name, 'wb') as f:#w: write; b: binary
        pickle.dump(dataset, f,  protocol = pickle.HIGHEST_PROTOCOL) #protocol 2 is the newest protocol, allows for efficient pickling of new-style classes
    print("Dataset written to file")

def read_set(dir_name, fname):
    '''
    Returns a list of unique items from an external file, for example, stopwords.
    @param fname: Name of file that should be read to create a list from
    @return list of terms in the file
    '''
    lst = set()
    for line in open(dir_name+fname):
        lst.add(line.strip())
    return list(lst)

def read_list(dir_name, fname):
    '''
    Returns a list of items from an external text file with one item per line.
    @param fname: Name of file that should be read to create a list from
    @return list of terms in the file
    '''
    lst = list()
    for line in open(dir_name+fname):
        lst.append(line.strip())
    return lst

def write_list(cur_list, out_dir, fname):
    '''
    Writes a list of values to a text file, such that each value is in a new line
    '''   
    with open(out_dir+fname,'w') as f:
        for term in cur_list:
            f.write(str(term)+'\n')
    
def read_data(path_dir, f_name, pickle_data = False, compress = True):
    '''
    Load any gzip pickled file (not necessarily in the dataset format)
    @param path_dir: path to the directory where file is located
    @param f_name: name of file on disc
    @return data object
    '''
    print("Loading data")
    if pickle_data:
        if compress:
            return pickle.load(gzip.open(path_dir+f_name))
        else:
            return pickle.load(open(path_dir+f_name, 'rb'))
    else:
        return json.load(open(path_dir+f_name, 'r'))
    print("Loaded data")

def get_dd_int():
    '''
    Returns a default dictionary of integer elements
    '''
    return defaultdict(int)


def create_labs_lexicon(dir_lexicon, molis_lexicon):
    '''
    Create a dictionary representation of different MOLIS lab test descriptions present in external file
    @param dir_lexicon: the directory which contains the lexicon of the MOLIS lab test descriptions
    @param molis_lexicon: file containing MOLIS lab test descriptions, one in each line, space replaced with '_'
    '''
    labs = {}
    with open(dir_lexicon+molis_lexicon) as f:
        for line in f:
            labs[line.lower().strip()] = len(line)
        
        labs = OrderedDict(sorted(labs.items(), key=operator.itemgetter(1), reverse = True))
#     labs = OrderedDict(list(labs).sort(key = len, reverse = True))
    return labs

def write_feat_importance(importance_score, importance_type, dir_out, task, vocab, vocab_link = True):
    """
    Write importance score for features for a given model.
    @param importance_score: score that indicates a feature importance value for the model. 
    @param importance_type: (recon_error/signficance) the type of importance that the score indicates. 
                            recon_error: mean squared reconstruction error for each output node in autoencoder, averaged across samples
                            significance: significance score of an input unit in a given network 
    @param dir_out: output directory
    @param task: task for which importance score is calculated
    @param vocab: feature vocabulary
    @param vocab_link: True to link back the feature number to the corresponding vocabulary item
    """
    if vocab_link:
        rev_vocab = dict()
        for k, v in vocab.items():
            rev_vocab[v] = k
    
    if importance_type is 'recon_error':
        #sort in ascending order, because lower loss is better
        imp_order = np.argsort(importance_score)
    else:
        #sort in descending order
        imp_order = np.argsort(-importance_score)
    
    importance = OrderedDict()
    for cur_feat in imp_order: 
        if vocab_link:
            importance[rev_vocab[cur_feat]] = str(importance_score[cur_feat])
        else:
            importance[cur_feat] = str(importance_score[cur_feat])
        
    #write features in the order most important for network, to least important
    write_data(importance, out_dir = dir_out, f_name = 'feat_importance_'+importance_type+'_'+task+'.json')
    
def write_feat_importance_class(importance_class, importance_type, dir_out, task, vocab, vocab_link = True):
    """
    Write importance score for features for a given model.
    @param importance_class: score that indicates a feature importance value for the each class. 
    @param importance_type (significance): significance score of an input unit wrt every output unit in a given network 
    @param dir_out: output directory
    @param task: task for which importance score is calculated
    @param vocab: feature vocabulary
    @param vocab_link: True to link back the feature number to the corresponding vocabulary item
    """
    if vocab_link:
        rev_vocab = dict()
        for k, v in vocab.items():
            rev_vocab[v] = k
    
    for cur_class in range(importance_class.shape[1]):
        importance_score = importance_class[:,cur_class]
        imp_order = np.argsort(-importance_score)
        
        importance = OrderedDict()
        for cur_feat in imp_order: 
            if vocab_link:
                importance[rev_vocab[cur_feat]] = str(importance_score[cur_feat])
            else:
                importance[cur_feat] = str(importance_score[cur_feat])
            
        #write features in the order most important for network, to least important
        write_data(importance, out_dir = dir_out, f_name = 'feat_importance_'+importance_type+'_'+task+'_'+str(cur_class)+'.json')