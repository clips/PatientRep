import numpy as np
from gensim.models import doc2vec
import multiprocessing as mp
from nlp_utils import preprocess_token

class Corpus(object):
    def __init__(self, dataset, pt_idx, dir_in, string_tags=False):
        
        self.string_tags = string_tags
        self.dataset = dataset
        self.pt_idx = pt_idx
        self.dir_in = dir_in
        
    def _tag(self, i):
        return i if not self.string_tags else '_*%d' % i

    def __iter__(self):
        for i, idx in enumerate(self.pt_idx):
            tokens = list()
            for cur_note in self.dataset.patients[idx].notes:
                with open(self.dir_in + str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f:
                    for cur_token in f.read().split():
                        tokens.append(preprocess_token(cur_token, lc = True, update_num = True, remove_punc = False, replace = False))
            yield doc2vec.TaggedDocument(tokens, [i])
                
class Doc2Vec(object):
    """
    Class to train document representation model using doc2vec
    Further get the training document vectors
    Infer vectors of unknown documents
    """
    def __init__(self, dim = 300, dm = 0, min_count = 10, ws = 3, ns = 5, n_iter = 5):
        '''
        @param dim: dimension of dense representation
        @param dm: 0 for distributed memory, 1 for dbow
        @param min_count: minimum frequency of term in corpus
        @param ws: size of context window
        @param ns: number of negative samples per positive sample
        @param n_iter: number of iterations to train model for
        '''
        n_cores = mp.cpu_count()
        self.dim = dim
        self.model = doc2vec.Doc2Vec(size = dim, dm = dm, min_count = min_count, window = ws, negative= ns, iter = n_iter, hs=0, workers = n_cores, seed = 1337)
        
    def train(self, dir_in, dir_out, dataset):
        """
        Build vocabulary and train a doc2vec model
        @param dir_in: Directory with normalized input documents
        @param dir_out: directory to save model in
        @param dataset: dataset object
        """
        corpus = Corpus(dataset, pt_idx = dataset.train_idx, dir_in = dir_in)
        
        print("Building vocab")
        self.model.build_vocab(corpus)
        
        print("Training")
        self.model.train(corpus)
        
        self.model.save(dir_out+'doc2vec_model_'+str(self.dim))
    
    def get_train_vec(self, dataset):
        """
        Return document vectors for training vectors
        Returns a 2D array of shape (num_of_train_inst, dim), where each index corresponds to instance number in dataset.train_idx
        @param dataset: dataset object
        """
        train_vec = np.zeros(shape = (len(dataset.train_idx), self.dim))
        for i, idx in enumerate(dataset.train_idx):
            train_vec[i,:] = self.model.docvecs[i]
            
        return train_vec
            
    def get_val_test_vec(self, dir_in, dataset):
        """
        Return vectors of documents in validation and test set
        @param dir_in: Directory containing normalized text
        @param dataset: dataset object
        """
        
        val_vec = self.get_subset_vec(dir_in, dataset, subset = 'val')
        test_vec = self.get_subset_vec(dir_in, dataset, subset = 'test')
        
        return val_vec, test_vec
     
    def get_subset_vec(self, dir_in, dataset, subset):
        """
        Return vectors of documents in a given data subset
        @param dir_in: Directory containing normalized text
        @param dataset: dataset object
        @param subset: (val/test) subset of data
        """
        if subset == 'val':
            n_inst = len(dataset.val_idx)
            pt_idx = dataset.val_idx
        elif subset == 'test':
            n_inst = len(dataset.test_idx)
            pt_idx = dataset.test_idx
            
        vec = np.zeros(shape = (n_inst, self.dim))
        
        for i, doc in enumerate(self.pt_data_generator(dir_in, dataset, pt_idx)):
            vec[i, :] = self.model.infer_vector(doc)
            
        return vec
            
    def get_vec(self, words):
        """
        Get the vector for a document given a list of words in the document
        @param words: List of words in a document
        """
        return self.model.infer_vector(words)
    
    def load_model(self, cfg):
        """
        Loads a pretrained doc2vec model and returns it
        """
        self.model = doc2vec.Doc2Vec.load(cfg.path_cfg.PATH_OUTPUT+'doc2vec_model_'+str(self.dim))
        print("Doc2vec model vocabulary size:", len(self.model.vocab))
        return self.model
    
    def pt_data_generator(self, dir_in, dataset, pt_idx):
        """
        Yields list of tokens in one document (patient data in our case) at a time
        @param dir_in: Path with normalized input documents
        @param dataset: dataset with all patient information
        @param pt_idx: patient indices for train/val/test subset
        """
        for i, idx in enumerate(pt_idx):
            tokens = list()
            for cur_note in dataset.patients[idx].notes:
                with open(dir_in + str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f:
                    for cur_token in f.read().split():
                        tokens.append(preprocess_token(cur_token, lc = True, update_num = True, remove_punc = False, replace = False))
            yield tokens