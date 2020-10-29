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
 
    mimic_data.load_feats_labels(cfg)

    fit_classifier = mimic_data.train_classifier(cfg)
    pred_score = mimic_data.predict(fit_classifier, cfg, test_type = 'test')
    
    utils.write_data(pred_score, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score.pkl.gz', pickle_data = True, compress = True)
    
    #pred_score = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score.pkl.gz', pickle_data = True, compress = True)
    
    y_pred = np.zeros(pred_score.shape[0], dtype = np.int32)
    for i in range(pred_score.shape[0]):
        y_pred[i] = np.argmax(pred_score[i])

    mimic_data.get_significant_feats(cfg, fit_classifier, y_pred, instance_type =  'test')

    score, conf_matrix, error_idx = mimic_data.evaluate(y_pred = y_pred, y_pred_score = pred_score, test_type='test')
    
    f = open(cfg.path_cfg.PATH_OUTPUT+'results_test.txt', 'w')
         
    f.write("Macro-averaged precision: %s, recall: %s and F-score: %s \n" % (score['prec_macro'], score['recall_macro'], score['f_score_macro']))
    f.write("Weighted precision: %s, recall: %s and F-score: %s \n" % (score['prec_weighted'], score['recall_weighted'], score['f_score_weighted']))
    f.write("Class based precision: %s \nClass based recall: %s \nClass based F-score: %s \n"
             % (score['prec_all'], score['recall_all'], score['f_score_all']))
    if 'auc' in score:
        f.write("AUC: %s \n" % (score['auc']))
        
    f.write("Confusion matrix: \n %s" % (conf_matrix))    
        
    f.close()

#     incorrect_pids = dict()
#     for cur_idx in error_idx:
#         incorrect_pids[mimic_data.patients[mimic_data.val_idx[cur_idx]].pid] = np.argmax(pred_score[cur_idx])
#     for cur_pt in mimic_data.patients:
#         if cur_pt.pid in incorrect_pids:
#             print(cur_pt.pid, "mortality class: ", cur_pt.mortality, incorrect_pids[cur_pt.pid])
    
if __name__ == '__main__':
    np.random.seed(1337)
    config_path = '../config/config.xml'
    if getattr( sys, 'frozen', False ):
        # running in a bundle
        config_path = '../../'+config_path
    main(config_path)
