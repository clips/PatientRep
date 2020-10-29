import numpy as np

from sdae import StackedDenoisingAE
from doc2vec import Doc2Vec
import utils
from nn_utils import load_model
from utils import write_feat_importance
import param_config_utils

class DenseRep(object):
    '''
    Class to generate and write dense representation of data
    '''
    def __init__(self, cfg, dataset, load_model, params):
        '''
        @param cfg: config object
        @param dataset: the dataset to generate dense representation of
        '''
        self.rep_type = cfg.run_cfg.rep_type
        if load_model:
            self.load_dense(cfg, dataset)
        else:
            self.train_dense_model(cfg, dataset, params)
        
    def train_dense_model(self, cfg, dataset, params):
        '''
        Train and write dense representation using required algorithms
        @param cfg: config object
        @param dataset: the dataset to generate dense representation of
        '''
        if self.rep_type == 'sdae':
            
            n_layers, n_hid, dropout = param_config_utils.get_sdae_params(cfg, params)
            
            cur_sdae = StackedDenoisingAE(n_layers = n_layers, n_hid = n_hid, dropout = dropout, nb_epoch = cfg.run_cfg.n_epochs)
            print("Training SDAE")
            
            self.model, (self.dense_train, self.dense_val, self.dense_test), node_recon_mse = cur_sdae.get_pretrained_sda(dataset.feats.feats_train, dataset.feats.feats_val, dataset.feats.feats_test, dir_out = cfg.path_cfg.PATH_OUTPUT)
            write_feat_importance(node_recon_mse, 'recon_error', cfg.path_cfg.PATH_OUTPUT, cfg.run_cfg.pred_type,  dataset.feats.feat_vocab_idx, vocab_link = True)

        elif self.rep_type == 'doc2vec':
            if params:
                doc2vec_inst = self._run_d2v_with_params(params)
            else:
                doc2vec_inst = Doc2Vec()
            if cfg.run_cfg.feat_type == 'bocui':
                dir_in = cfg.path_cfg.PATH_CUI_MIMIC
            else:
                dir_in = cfg.path_cfg.PATH_NORMALIZED_MIMIC
            
            doc2vec_inst.train(dir_in, cfg.path_cfg.PATH_OUTPUT, dataset)
            
            self.model = doc2vec_inst.model
            
            self.dense_train = doc2vec_inst.get_train_vec(dataset)
            self.dense_val, self.dense_test = doc2vec_inst.get_val_test_vec(cfg.path_cfg.PATH_NORMALIZED_MIMIC, dataset)
            
        self._write_dense(cfg)
            
    def load_dense(self, cfg, dataset):
        """
        Load pretrained model for dense representation
        """
        if self.rep_type == 'sdae':
            self.model = load_model(cfg.path_cfg.PATH_OUTPUT, f_model = 'enc_layers.json', f_weights = 'enc_layers_weights.h5')
            try:    
                self._read_dense(cfg)
            except:
                print("Did not find dense pretrained data")
        elif self.rep_type == 'doc2vec':
            doc2vec_inst = Doc2Vec()
            self.model = doc2vec_inst.load_model(cfg)
            try:    
                self._read_dense(cfg)
            except:
                self.dense_train = doc2vec_inst.get_train_vec(dataset)
                self.dense_val, self.dense_test = doc2vec_inst.get_val_test_vec(cfg.path_cfg.PATH_NORMALIZED_MIMIC, dataset)
                self._write_dense(cfg)
            
    def _write_dense(self, cfg):
        '''
        Write train, val, and test dense datasets to file
        '''
        if self.rep_type == 'doc2vec':
            prefix = 'doc2vec_'
        elif self.rep_type == 'sdae':
            prefix = 'sdae_'
        utils.write_data(self.dense_train, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_train.pkl.gz', pickle_data= True, compress = True)
        utils.write_data(self.dense_val, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_val.pkl.gz', pickle_data= True, compress = True) 
        utils.write_data(self.dense_test, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_test.pkl.gz', pickle_data= True, compress = True) 
        
    def _read_dense(self, cfg):
        '''
        Read train, val, and test dense datasets from file
        '''
        if self.rep_type == 'doc2vec':
            prefix = 'doc2vec_'
        elif self.rep_type == 'sdae':
            prefix = 'sdae_'
        self.dense_train = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_train.pkl.gz', pickle_data= True, compress = True)
        self.dense_val = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_val.pkl.gz', pickle_data= True, compress = True) 
        self.dense_test = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = prefix+'dense_test.pkl.gz', pickle_data= True, compress = True) 
        
    def _write_csv(self, cfg):
        
        if self.rep_type == 'doc2vec':
            prefix = 'doc2vec_'
        elif self.rep_type == 'sdae':
            prefix = 'sdae_'
            
        np.savetxt(fname = cfg.path_cfg.PATH_OUTPUT + prefix + 'dense_train.csv.gz', X = self.dense_train, delimiter = ',')
        np.savetxt(fname = cfg.path_cfg.PATH_OUTPUT + prefix + 'dense_val.csv.gz', X = self.dense_val, delimiter = ',')
        np.savetxt(fname = cfg.path_cfg.PATH_OUTPUT + prefix + 'dense_test.csv.gz', X = self.dense_test, delimiter = ',')    
    
    def _run_d2v_with_params(self, params):
        '''
        Train/load doc2vec model under given non-default parameter settings
        '''
        dim = params["dimension"]
        win_size = params["win_size"]
        min_freq = params["min_freq"]
        neg_samples = params["neg_samples"]
        dm = params["dm"]
        
        return Doc2Vec(dim = dim, ws=win_size, ns=neg_samples, dm = dm, min_count=min_freq)
