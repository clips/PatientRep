import os
import re

import nlp_utils
import config

def normalize_text(dir_tokenized, dir_out, replace):
    """
    Normalize and preprocess tokens and write them to the file
    """
    for file in os.scandir(dir_tokenized):
        if 'txt' in file.name:
            with open(dir_out + file.name, 'w') as f_out:
            
                for line in open(dir_tokenized+file.name):
                    line = line.split()
                    
                    for i, token in enumerate(line):
                        line[i] = nlp_utils.preprocess_token(token, lc = True, update_num = True, remove_punc = False, fix_labs = False, labs = None, replace = replace)
                    
                    if '' in line:
                        line.remove('')
                    line = ' '.join(line)
                    
                    line = re.sub(' +',' ',line)
                    f_out.write(line+'\n')
            
if __name__ == '__main__':
    cfg_file = '../config/config.xml'
    cfg = config.Config(cfg_file)
    normalize_text(cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC, cfg.path_cfg.PATH_NORMALIZED_MIMIC, replace = False)
