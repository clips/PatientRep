from glob import iglob
import os

import config

def write_cui(dir_clamp, dir_out):
    """
    Rewrite the content of the notes such that concepts are treated as a single token
    @param dir_tokenized: directory containing tokenized notes
    @param dir_clamp: directory containing the preprocessed output using clamp
    @param dir_out: directory to write the new content in
    """
    
    
    for filename in iglob(dir_clamp+'*.txt'):
        with open(filename, 'r') as f_clamp:
            new_content = ''
    
            for line in f_clamp:
                    
                line = line.split('\t')
                    
                if line[0] != 'NamedEntity':
                    break #assuming that named entities are the first lines of the file (to speed up the process)
                
                cur_cui = ''
                    
                for i in range(3,len(line),1):
                    if line[i].startswith('assertion'):
                        cur_assertion = line[i].split('=')[1]
                    elif line[i].startswith('cui') :
                        cur_cui = line[i].split('=')[1]
                    else:
                        continue
                
                new_content += cur_cui + '_' + cur_assertion + ' '
                 
            with open(dir_out + os.path.basename(filename), 'w') as f_out:    
                f_out.write(new_content)
                        
if __name__ == '__main__':
    cfg_file = '../config/config.xml'
    cfg = config.Config(cfg_file)
    write_cui(cfg.path_cfg.PATH_PROCESSED_CLAMP_MIMIC, cfg.path_cfg.PATH_CUI_MIMIC)
            
            