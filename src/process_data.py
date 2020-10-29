'''
@author madhumita
'''

import subprocess
import os
#import ucto

class Process_Data(object):
    '''
    If the texts have not been processed already, run an NLP pipeline to process them.
    Thereby read the processed annotations and store them in the desired format.
    '''
    
    def __init__(self, cfg):
        '''
        @param cfg: Config object
        '''
        if not cfg.run_cfg.processed:
            self.process_texts(cfg)

            
    def process_texts(self, cfg):
        '''
        First, performs sentence splitting and tokenization using UCTO
        Thereby, performs concept identification if required using a lang dependent pipeline
        @param cfg: Config object
        '''
#         if cfg.run_cfg.p_type == 'clamp':
#             subprocess.call([cfg.run_cfg.pipeline])
#         elif cfg.run_cfg.p_type == 'ucto':
        self.preprocess_ucto(cfg) #sentence splitting and tokenization using UCTO
        #add concept identification using clamp
                        
    def preprocess_ucto(self, cfg, skip_dup_token = True, folia_out = False):
        '''
        Performs sentence splitting and tokenization using UCTO and saves the tokenized data as a separate file
        @param cfg: config object
        @param skip_dup_token: True to skip tokenizing file that is already tokenized
        @param folia_out: True to write output in folia format.
        '''
        if cfg.run_cfg.lang == 'EN':
            config = "tokconfig-eng"
            path_content = cfg.path_cfg.PATH_CONTENT_MIMIC
            path_processed = cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC
        
        cur_tokenizer = ucto.Tokenizer(config, sentenceperlineinput = False, sentenceperlineoutput = True, sentencedetection = True, foliaoutput= folia_out)
        
        for cur_file in os.listdir(path_content):
            if not skip_dup_token or not os.path.isfile(path_processed+cur_file):
                cur_tokenizer.tokenize(path_content+cur_file, path_processed+cur_file)
                
#         for cur_file in os.scandir(path_content):
#             if not skip_dup_token or not os.path.isfile(path_processed+cur_file.name):
#                 cur_tokenizer.tokenize(path_content+cur_file.name, path_processed+cur_file.name)
            
        
        
