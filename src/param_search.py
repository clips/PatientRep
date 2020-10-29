import os    
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"
os.environ['KERAS_BACKEND'] = "theano"
os.environ['PYTHONHASHSEED'] = '0' 

import gc
import numpy as np

from randomized_hyperparam_search import RandomizedSearch, doc2vec_parameter_space, sdae_parameter_space, ffnn_parameter_space
from config import Config
from utils import read_data, write_data

def train_and_eval_models(mimic_data, params):
    if search_space == 'ffnn':
        mimic_data.load_feats_labels(cfg, params = None)
        fit_classifier = mimic_data.train_classifier(cfg, params)
    else:
        mimic_data.load_feats_labels(cfg, params)
        if search_space == 'sdae_finetune':
            fit_classifier = mimic_data.train_classifier(cfg, params = params)
        else:
            fit_classifier = mimic_data.train_classifier(cfg, params = None)
    
    
    pred_score = mimic_data.predict(fit_classifier, cfg, test_type = 'val')
    score, _, _ = mimic_data.evaluate(y_pred = None, y_pred_score = pred_score, test_type='val')
    
    del fit_classifier
    for i in range(3): gc.collect()
    
    del score['prec_all']
    del score['recall_all']
    del score['f_score_all']
        
    print(params)
    print(score)
    
    return score


def get_best_params(all_scores, search_space):
    best_scores = {}
    best_runs = {}
    for eval_method in all_scores[0].keys():
        best_scores[eval_method] = 0

    for n_run, results in all_scores.items():
        for eval_method, score in results.items():
            if score > best_scores[eval_method]:
                best_scores[eval_method] = score
                best_runs[eval_method] = n_run
    return best_runs, best_scores



if __name__ == "__main__":
    n_times = 50
    
    search_space = 'ffnn' #doc2vec/sdae_pretrain/sdae_finetune/ffnn
    cfg = Config('../config/'+search_space.split('_')[0]+'_param_search_cfg.xml')
    np.random.seed(cfg.run_cfg.seed)

    mimic_data = read_data(path_dir = cfg.path_cfg.PATH_INPUT, f_name = 'mimiciii_with_labels.pkl.gz', pickle_data = True, compress = True)
#     mimic_data.load_feats_labels(cfg)
    
    all_params = {}
    all_scores = {}
    
    if search_space == 'doc2vec':
        param_space = doc2vec_parameter_space
    elif search_space.startswith('sdae'):
        param_space = sdae_parameter_space
    elif search_space == 'ffnn':
        param_space = ffnn_parameter_space
        
    random_search = RandomizedSearch(param_space)
    for n in range(n_times):
        print("Run {}...".format(n))
        params = random_search.sample()
        score = train_and_eval_models(mimic_data, params)
        
        all_params[n] = params
        all_scores[n] = score
    
    best_runs, best_scores = get_best_params(all_scores, search_space)

    out_dir = '../output/'+search_space+'_rand_search/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    write_data(best_runs, out_dir, f_name='best_runs.json')
    write_data(best_scores, out_dir, f_name='best_scores.json')
    write_data(all_params, out_dir, f_name='all_params.json')
    write_data(all_scores, out_dir, f_name='all_scores.json')

    for eval_method, n_run in best_runs.items():
        print("{}: {}={}".format(eval_method, n_run, all_params[n_run]))
