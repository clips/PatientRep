import numpy as np

from config import Config
from utils import read_data, write_data
from nlp_utils import preprocess_token

def get_vocab_corpus_freq(cfg, dataset):
    corpus_freq = dict()
    
    feat_vocab = read_data(cfg.path_cfg.PATH_INPUT, f_name='feat_vocab_' + cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
    for term in feat_vocab.keys():
        corpus_freq[term] = 0
        
    for cur_idx in dataset.train_idx:
        for cur_note in dataset.patients[cur_idx].notes:
            with open(cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f_proc:
                        
                for cur_sent in f_proc:
                    cur_sent = cur_sent.strip()
                    for token_str in cur_sent.split(' '):
                        token_str = preprocess_token(token_str, lc = True, update_num = True, remove_punc = True, replace = True, fix_labs = False, labs = None)
                    
                        if token_str in corpus_freq:
                            corpus_freq[token_str] += 1
                            
                            
    write_data(data = corpus_freq, out_dir = cfg.path_cfg.PATH_INPUT, f_name = 'train_corpus_freq_'+ cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
    
if __name__ == '__main__':
    cfg = Config("../config/config.xml")
    np.random.seed(cfg.run_cfg.seed)
    mimic_data = read_data(path_dir = cfg.path_cfg.PATH_INPUT, f_name = 'mimiciii_with_labels.pkl.gz', pickle_data = True, compress = True)
    
    get_vocab_corpus_freq(cfg, mimic_data)