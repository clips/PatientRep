import fasttext
import re
import os

class Embedding(object):
    def __init__(self):
        '''
        @param dim: the number of dimensions to generate the vector of
        @param algo (sg/cbow): the training algorithm for vector generation
        @param tool (fasttext): tool to train vectors with 
        '''
        self.model = None
        
    def gen_embeddings(self, dim, dir_in, f_in, dir_model, f_model, algo = 'sg', tool = 'fasttext'):
        '''
        Generate word embeddings
        @param cfg: Config object
        @param f_in: input text file to generate embeddings from
        @param f_model: output model file
        '''
        f_norm = f_in.split('.')[0]+'_norm.txt'
        if not os.path.exists(dir_in+f_norm):
            print("Normalizing text before generating embeddings")
            self._normalize_text(dir_in = dir_in, f_in = f_in, dir_out = dir_in, f_out = f_norm, lc = True, replace_num = True)
        
        if tool == 'fasttext': 
            #find tool desc here: https://pypi.python.org/pypi/fasttext
            if algo == 'sg':
                self.model = fasttext.skipgram(dir_in+f_norm, dir_model + 'ft_model_sg_'+f_model+'_'+str(dim), dim = dim) #skipgram
            elif algo == 'cbow':
                self.model = fasttext.cbow(dir_in+f_norm, dir_model + 'ft_model_cbow_'+f_model+'_'+str(dim), dim = dim) #cbow
            
    def load_embedding_model(self, dir_model, f_model, dim, algo = 'sg', tool = 'fasttext'):
        '''
        Load a pretrained word embedding model
        @param cfg: Config object
        @param f_model: the pretrained model
        @param algo: (sg/cbow) skip-gram or cbow
        @param tool: (fasttext) Tool for training the embeddings
        '''
        if tool == 'fasttext':
            if algo == 'sg':
                self.model = fasttext.load_model(dir_model+'ft_model_sg_'+f_model+'_'+str(dim)+'.bin')
            elif algo == 'cbow':
                self.model = fasttext.load_model(dir_model+'ft_model_cbow_'+f_model+'_'+str(dim)+'.bin')
        
        vocab = self.model.words
        print("Total vocabulary:", len(vocab))
        
        
    def _normalize_text(self, dir_in, f_in, dir_out, f_out, lc = True, replace_num = True):
        """
        Normalize input text before generating vectors
        @param dir_in: input directory
        @param f_in: input file
        @param dir_out: output directory
        @param f_out: normalized file
        @param lc: True to lowercase text
        @param replace_num: True to replace numbers with 'N'
        """
        
        pattern_num = re.compile("^[+-]?[0-9]*[.,]?[0-9]+$") #regex for numeric values
        
        with open(dir_out+f_out, 'w') as f_out:
            for line in open(dir_in+f_in):
                if lc:
                    line = line.lower()
                for cur_token in line.strip().split():
                    if replace_num and pattern_num.match(cur_token):
                        line = line.replace(cur_token, 'N')
                f_out.write(line)
                
