'''
@author madhumita
'''

import subprocess

class Process_Eng(object):
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
        Run the required NLP pipeline on the input texts
        @param cfg: Config object
        '''
        if cfg.run_cfg.p_type == 'clamp':
            subprocess.call([cfg.run_cfg.pipeline])