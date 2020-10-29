import os
import config

def replace_concepts(dir_tokenized, dir_clamp, dir_out):
    """
    Rewrite the content of the notes such that concepts are treated as a single token
    @param dir_tokenized: directory containing tokenized notes
    @param dir_clamp: directory containing the preprocessed output using clamp
    @param dir_out: directory to write the new content in
    """
    for file in os.scandir(dir_tokenized):
        if 'txt' in file.name:
            content = open(dir_tokenized + file.name, 'r').read()
            
            with open(dir_clamp + file.name, 'r') as f_clamp:
                for line in f_clamp:
                    line = line.split('\t')
                    
                    if line[0] != 'NamedEntity':
                        break #assuming that named entities are the first lines of the file (to speed up the process)
                    
                    start = int(line[1])
                    end = int(line[2])
                    
#                     for i in range(3,len(line),1):
#                         if not line[i].startswith('ne'):
#                             continue
#                         else:
#                             concept_txt = line[i].split('=')[1]
#                             break
#                         
#                     print(concept_txt)
                    content = content[:start] + content[start:end].replace(" ", "_") + content[end:]
                    
                with open(dir_out + file.name, 'w') as f_out:  
                    f_out.write(content)
            
if __name__ == '__main__':
    cfg_file = '../config/config.xml'
    cfg = config.Config(cfg_file)
    replace_concepts(cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC, cfg.path_cfg.PATH_PROCESSED_CLAMP_MIMIC, cfg.path_cfg.PATH_CONTENT_CONCEPTS_MIMIC)
            
            