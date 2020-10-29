import os    
os.environ['THEANO_FLAGS'] = "device=gpu1,floatX=float32"
os.environ['KERAS_BACKEND'] = "theano"
os.environ['PYTHONHASHSEED'] = '0' 

import numpy as np
import sys
import config
import utils
from process_data import Process_Data
import mimiciii as mimic
from embedding import Embedding

from pca import PCA

def main(cfg_file):
    '''
    Pipeline to identify patient mortality on MIMIC dataset
    '''
    cfg = config.Config(cfg_file)
    
    np.random.seed(cfg.run_cfg.seed)

    if cfg.run_cfg.init_data:
        mimic_data = mimic.MIMICIII(cfg)
    else:
        mimic_data = utils.read_data(path_dir = cfg.path_cfg.PATH_INPUT, f_name = 'mimiciii_with_labels.pkl.gz', pickle_data = True, compress = True)
    mimic_data.write_vis_data(cfg)
        
    mimic_data.cluster(cfg)
      
    Process_Data(cfg) #process the texts with an NLP pipeline
    
    
    mimic_data.get_text_stats(cfg)
    
    mimic_data.write_all_train_tokenized(cfg) #write space delimited tokens for all training notes to train fasttext vectors
 
    if cfg.run_cfg.process_embeddings == 'gen':
        mimic_data.gen_embeddings(cfg)
    elif cfg.run_cfg.process_embeddings == 'load':
        mimic_data.load_embeddings(cfg)

  
    mimic_data.load_feats_labels(cfg)

#
    # cur_pca = PCA(mimic_data.feats, mimic_data.y_train)
    # mimic_data.write_vis_data(cfg)

    if cfg.run_cfg.select_best_feats:
        mimic_data.select_best_feats(cfg)

    fit_classifier = mimic_data.train_classifier(cfg)
    pred_score = mimic_data.predict(fit_classifier, cfg, test_type = 'val')
       
    utils.write_data(pred_score, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'val_pred_score.pkl.gz', pickle_data = True, compress = True)
         
    pred_score = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'val_pred_score.pkl.gz', pickle_data = True, compress = True)
      
    score, conf_matrix, error_idx = mimic_data.evaluate(y_pred = None, y_pred_score = pred_score, test_type='val')
    
    f = open(cfg.path_cfg.PATH_OUTPUT+'results.txt', 'w')
         
    f.write("Macro-averaged precision: %s, recall: %s and F-score: %s \n" % (score['prec_macro'], score['recall_macro'], score['f_score_macro']))
    f.write("Weighted precision: %s, recall: %s and F-score: %s \n" % (score['prec_weighted'], score['recall_weighted'], score['f_score_weighted']))
    f.write("Class based precision: %s \nClass based recall: %s \nClass based F-score: %s \n"
             % (score['prec_all'], score['recall_all'], score['f_score_all']))
    if 'auc' in score:
        f.write("AUC: %s \n" % (score['auc']))
        
    f.write("Confusion matrix: \n %s" % (conf_matrix))
    f.close()


if __name__ == '__main__':
    np.random.seed(1337)
    config_path = '../config/config.xml'
    if getattr( sys, 'frozen', False ):
        # running in a bundle
        config_path = '../../'+config_path
    main(config_path)
