from collections import defaultdict, OrderedDict
import operator
import numpy as np
from zipfile import ZipFile
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.sparse.base import issparse
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing.data import Normalizer, StandardScaler, MinMaxScaler

# import nlp_utils
import utils 


class Features(object):
        
    def __init__(self, featurize, dataset, feat_type, val_type, cfg):
        '''
        @param dataset: object of the dataset, for example of type MIMICIII
        @param feat_type (bow): Type of features
        '''
        self.feat_type = feat_type.lower()
        self.val_type = val_type.lower()
        
        if featurize:
        
            self.vocab = None #to be updated while generating training features
            
            if self.feat_type == 'bow' or self.feat_type == 'boconcept' or self.feat_type == 'bocui':
                
                train_feats = self._get_feat_bag(cfg, dataset, dataset.train_idx, train = True)
                val_feats = self._get_feat_bag(cfg, dataset, dataset.val_idx, train = False)
                test_feats = self._get_feat_bag(cfg, dataset, dataset.test_idx, train = False)
            
            self._vectorize_feats(cfg, train_feats, val_feats, test_feats)
            
            self.get_vocab_stats()
            
            utils.write_data(self.feats_train, cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_train.pkl.gz', pickle_data=True, compress=True)
            utils.write_data(self.feats_val, cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_val.pkl.gz', pickle_data=True, compress=True)
            utils.write_data(self.feats_test, cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_test.pkl.gz', pickle_data=True, compress=True)
            utils.write_data(self.feat_vocab_idx, out_dir = cfg.path_cfg.PATH_INPUT, f_name = 'feat_vocab_'+self.feat_type+ '_' + cfg.run_cfg.content_type+ '_' + cfg.run_cfg.val_type+'.json')
        
        else:
#             self.feats_train = self.feats_val = self.feats_test = None
            self.feats_train = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_train.pkl.gz', pickle_data=True, compress=True)
            self.feats_val = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_val.pkl.gz', pickle_data=True, compress=True)
            self.feats_test = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_test.pkl.gz', pickle_data=True, compress=True)
             
            self.feat_vocab_idx = utils.read_data(cfg.path_cfg.PATH_INPUT, f_name='feat_vocab_' + cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
            self.feat_freq = utils.read_data(cfg.path_cfg.PATH_INPUT, f_name = 'train_corpus_freq_'+ cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
            
            print("Vocabulary size:", len(self.feat_vocab_idx))   
    
    def _get_feat_bag(self, cfg, dataset, idx, train):
        '''
        Returns bag of tokens for a subset of the data (train/val/test) in the format: [{token:freq},...] in the same order as instance IDs.
        @param cfg: config object
        @param dataset: Dataset object, example, MIMICIII
        @param idx: indices of the instances in the dataset for the desired subset
        @train: True if training subset, False otherwise
        '''
        
        if cfg.run_cfg.lang == 'NL':
            path_processed = cfg.path_cfg.PATH_PROCESSED_UZA
            path_concept = None #@TODO: UPDATE!
            excl_tokens = {'<utt>', 'TS_U_BRIEF', 'TS_U_OKVERSLAG', 'TS_U_PROTOCOL', 'TS_U_NOTA', 'TS_U_OTHERS'}
        elif cfg.run_cfg.lang == 'EN':
            if cfg.run_cfg.content_type == 'concept':
                path_processed = cfg.path_cfg.PATH_CONTENT_CONCEPTS_MIMIC
            elif cfg.run_cfg.content_type == 'cui':
                path_processed = cfg.path_cfg.PATH_CONTENT_CUI_MIMIC
            else:
                path_processed = cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC
            path_concept = cfg.path_cfg.PATH_PROCESSED_CLAMP_MIMIC
            excl_tokens = set()
        
        if train:
            self.vocab = dict(list()) # initially {token: [corpus freq, set of pt token occurs for]} (contains corpus freq and DF counts). Updated later
            self.vocab['OOV'] =  [0,set()]
        
        feat_bag = self._get_feat_bag_pt(dataset, idx, train, path_processed, path_concept, excl_tokens)
            
        if train:
            self._update_vocab(len(idx), freq_th = 5, cfg = cfg)
        
        if self.val_type == 'tf':
            feat_bag = nlp_utils.get_tf(term_freq = feat_bag, idf = self.vocab)
        elif self.val_type == 'tf-idf':
            feat_bag = nlp_utils.get_tf_idf(term_freq = feat_bag, idf = self.vocab)
        else:
            print("Please enter a valid type of feature values: (tf/tf-idf)")
        
        return feat_bag
    
    def _get_feat_bag_pt(self, dataset, idx, train, path_processed, path_concept, excl_tokens):
        """
        Get the token bag with each patient as an instance
        @param dataset: the patient dataset
        @param idx: indices for the patients for the current subset of the dataset
        @param train: 1 if train subset
        @param path_processed: path for tokenized data
        @param excl_tokens: set of tokens to exclude from the bag of words
        """
        feat_bag = list(dict()) #[token:token_freq] for every instance 
        
        for i, cur_idx in enumerate(idx):
            
            if not i%100:
                print("Processing ", i," of ", len(idx), " patients")
                
            cur_feat_dict = defaultdict(int)
            
            for cur_note in dataset.patients[cur_idx].notes:
                
                if self.feat_type == 'bow':
                    cur_feat_dict = self._update_token_dict(path_processed, cur_note, excl_tokens, train, cur_idx, cur_feat_dict)
                elif self.feat_type == 'boconcept':
                    cur_feat_dict = self._update_concept_dict(path_concept, cur_note, excl_tokens, train, cur_idx, cur_feat_dict)
                elif self.feat_type == 'bocui':
                    cur_feat_dict = self._update_cui_dict(path_concept, cur_note, excl_tokens, train, cur_idx, cur_feat_dict)
                    
            if len(cur_feat_dict) == 0:
                print("Patient with empty features, ", dataset.patients[cur_idx].pid) #just a sanity check that we do not generate empty feats
            
            feat_bag.append(cur_feat_dict)  
        
        return feat_bag
    
    def _update_token_dict(self, path_processed, cur_note, excl_tokens, train, inst_idx, cur_token_dict):
        """
        Update token dictionary with the tokens in the current note
        @param path_processed: path containing tokenized data
        @param cur_note: current note object
        @param excl_tokens: set of tokens to exclude from consideration
        @param train: 1 if train set
        @param inst_idx: index of the current instance
        @param cur_token_dict: dictionary containing existing tokens and corresponding frequency 
                               for the current instance before the current note is processed (pt/notes)
        @return updated cur_token_dict
        """
        
#         f_proc = open(path_processed+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.rowid)+'.txt')
        zip_dir = path_processed[:path_processed.index('.zip')+4]
        subdir = path_processed[path_processed.index('.zip')+5:]

        with ZipFile(zip_dir) as basezip:
            with basezip.open(subdir+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f_proc:

                for cur_sent in f_proc:
                    cur_sent = cur_sent.strip()
                    for token_str in cur_sent.split(' '):
                        token_str = nlp_utils.preprocess_token(token_str, lc = True, remove_punc= True, replace = True, update_num = True, fix_labs = False, labs = None)

                        if token_str != '' and token_str not in excl_tokens:
                            if train:
                                self._update_vocab_item(token_str, inst_idx)
                            elif token_str not in self.vocab:
                                token_str = 'OOV'

                            cur_token_dict[token_str] += 1
        
        return cur_token_dict
    
    def _update_concept_dict(self, path_concept, cur_note, excl_tokens, train, inst_idx, cur_concept_dict):
        
#         f_proc = open(path_processed+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt')
        with open(path_concept+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f_clamp:
        
            for line in f_clamp:
                    
                line = line.split('\t')
                    
                if line[0] != 'NamedEntity':
                    break #assuming that named entities are the first lines of the file (to speed up the process)
                
    #             start = int(line[1])
    #             end = int(line[2])
                concept_str = ''    
                for i in range(3,len(line),1):
                    if line[i].startswith('assertion'):
                        cur_assertion = line[i].split('=')[1]
                    elif line[i].startswith('ne') :
                        concept_str = line[i].split('=')[1]
                    else:
                        continue
                
                cur_feat = concept_str + '_' + cur_assertion
                
                if cur_feat != '' and cur_feat not in excl_tokens:
                    
                    if train:
                        self._update_vocab_item(concept_str+'_present', inst_idx, cur_assertion == 'present')
                        self._update_vocab_item(concept_str+'_absent', inst_idx, cur_assertion == 'absent')
                    elif cur_feat not in self.vocab:
                        cur_feat = 'OOV'
                        
                    cur_concept_dict[cur_feat] += 1
                
        
        return cur_concept_dict
    
    def _update_cui_dict(self, path_concept, cur_note, excl_tokens, train, inst_idx, cur_cui_dict):

        zip_dir = path_concept[:path_concept.index('.zip') + 4]
        zipname = path_concept[path_concept.rfind('/', 0, path_concept.index('.zip'))+1:path_concept.index('.zip')]
        subdir = zipname+path_concept[path_concept.index('.zip') + 4:]

        with ZipFile(zip_dir) as basezip:
            with basezip.open(subdir+ str(cur_note.pid) + '_' + str(cur_note.hadmid) + '_' + str(cur_note.noteid) + '.txt') as f_clamp:

        # with open(path_concept+str(cur_note.pid)+'_'+str(cur_note.hadmid)+'_'+str(cur_note.noteid)+'.txt') as f_clamp:
        
                for line in f_clamp:

                    print(line)
                    print(type(line))

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

                    cur_feat = cur_cui + '_' + cur_assertion

                    if cur_feat != '' and cur_feat not in excl_tokens:

                        if train:
                            self._update_vocab_item(cur_cui+'_present', inst_idx, cur_assertion == 'present')
                            self._update_vocab_item(cur_cui+'_absent', inst_idx, cur_assertion == 'absent')
                        elif cur_feat not in self.vocab:
                            cur_feat = 'OOV'

                        cur_cui_dict[cur_feat] += 1
        
        return cur_cui_dict
            
    def _update_vocab_item(self, entry_str, idx, present = True):
        '''
        Update the occurrence of term in the vocabulary
        @param entry_str: vocab item string
        @param idx: instance ID for mention of the current item
        '''
        val = self.vocab.get(entry_str, [0,set()])
        if present:
            val[0] += 1 #corpus freq of term
            val[1].add(idx) #unique identifier for instance to calculate document freq of cur_token (how many different instances does it occur for)
        
        self.vocab[entry_str] = val 
    
    
    def _update_vocab(self, tot_inst, freq_th, cfg):
        '''
        Update the vocab by removing low frequency terms, and calculating IDF scores
        @param tot_inst: total number of instances in the training set
        @param freq_th: threshold for corpus freq of vocab. All items below this freq will be removed.
        '''
        self.vocab = nlp_utils.remove_lf_terms(self.vocab, freq_th)
        
        self._serialize_corpus_freq(cfg)
        
        self.vocab = nlp_utils.get_df(self.vocab) #{vocab_term:document_freq}
        self.vocab = nlp_utils.calculate_idf(self.vocab, ndocs = tot_inst)  #{vocab_term:inverse_document_freq}
    
    def _serialize_corpus_freq(self, cfg):
        '''
        Write frequency of terms in vocabulary in json format
        '''
        corpus_freq = dict()
        for cur_term, val in self.vocab.items():
            corpus_freq[cur_term] = val[0]
            
        utils.write_data(corpus_freq, out_dir = cfg.path_cfg.PATH_INPUT, f_name = 'train_corpus_freq_'+ cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
    
    def _vectorize_feats(self, cfg, train_feat_bag, val_feat_bag, test_feat_bag, data_type = np.float32):
        '''
        Generates sparse (scipy csr matrix) matrix for input features
        @param cfg: Config object
        @param train_feat_bag: training feats in the format: [{feat:value},...] in the same order as instance IDs.
        @param val_feat_bag: val feats in the format: [{feat:value},...] in the same order as instance IDs.
        @param test_feat_bag: test feats in the format: [{feat:value},...] in the same order as instance IDs.
        @param data_type: np.float32/np.float64 data type of generated features
        '''
        vectorizer = DictVectorizer(sparse = True, dtype = data_type)
        self.feats_train = vectorizer.fit_transform(train_feat_bag)
        self.feat_vocab_idx = vectorizer.vocabulary_
        
        self.feats_val = vectorizer.transform(val_feat_bag)
        self.feats_test = vectorizer.transform(test_feat_bag)
        
    
    def get_vocab_stats(self):
        '''
        Print statistics about the vocabulary, like how many terms are there, and the terms with the highest and the lowest IDF scores
        '''
        #sort vocabulary according to IDF score in decreasing order (lowest scores: most common words in corpus)
        self.vocab = OrderedDict(sorted(self.vocab.items(), key = operator.itemgetter(1), reverse = True))
        
        print("Total no. of vocab terms: ", len(self.vocab))
        print("Highest IDF score: ", list(self.vocab.items())[0])
        print("Lowest IDF score: ", list(self.vocab.items())[len(self.vocab) - 1])
        
    def select_feats(self, y_train, n = 100, n_feats_type = 'num', criteria = 'val', val_type = 'tf-idf'):
        '''
        Get the indices of the top features based on log TF IDF feature values in train set
        @param n: number (up to total num of features present), or proportion (between 0-1) of features to return
        @param n_feats_type (prop/num): whether 'n' is a proportion or a number of features
        '''
        
        if n_feats_type.lower() == 'prop':
            num_feats = int(n*self.feats_train.shape[1])
        else:
            assert n <= self.feats_train.shape[1], "Cannot select more features than originally present"
            num_feats = int(n)
        
        print("Selecting %s features of total %s feats" % (num_feats, self.feats_train.shape[1]))
        
        if criteria == 'val':
            (best_feats_train, best_feats_val, best_feats_test) = self._get_best_feats_val(num_feats)
            
        elif criteria == 'chi2':
            (best_feats_train, best_feats_val, best_feats_test) = self._get_best_feats_chi2(num_feats, y_train)
        
        self._write_best_feats(num_feats, val_type, criteria)
        
        return (best_feats_train, best_feats_val, best_feats_test)
        
    def _write_best_feats(self, k, val_type, criteria):
        """
        Write the best features to external file
        """
        rev_vocab = dict()
        for k, v in self.feat_vocab_idx.items():
            rev_vocab[v] = k
        f = open('../output/best_pt_feats_'+str(k)+'_'+val_type+'_'+criteria+'.txt', 'w')
        for i in range(len(self.top_feat_indices)):
            f.write(rev_vocab[self.top_feat_indices[i]]+'\n')
        f.close()
        
    def _get_best_feats_val(self, k):
        """
        Selects top k features for each instance based on feature values
        """
        top_feat_indices = set()
        for cur_inst in range(self.feats_train.shape[0]):
            cur_feats = self.feats_train.getrow(cur_inst).todense()
            cur_feats = np.array(cur_feats).reshape(-1,)
            
            cur_top_feat_idx = np.argpartition(-cur_feats, k)[:k] #returns top indices in random order, not sorted order
            
            for i in range(cur_top_feat_idx.shape[0]):
                if cur_feats[cur_top_feat_idx[i]]: #add a feature as top feature only if non zero. Assuming that 0 is vectorized value of feat absent in the instance
                    top_feat_indices.add(cur_top_feat_idx[i])
           
        print("Found total %s features across all instances" %len(top_feat_indices))
        
        self.top_feat_indices = np.array(list(top_feat_indices))
        return self.get_feats_from_indices()    
            
    def _get_best_feats_chi2(self, k, y_train):
        """
        Select top k features based on chi2 test of independence for given labels.
        """
            
        selector = SelectKBest(chi2, k)
        best_feats_train = selector.fit_transform(self.feats_train, y_train)
        self.top_feat_indices = selector.get_support(indices=True)
        best_feats_val = selector.transform(self.feats_val)
        best_feats_test = selector.transform(self.feats_test) 
        
        return (best_feats_train, best_feats_val, best_feats_test)
           
    def get_feats_from_indices(self):
        '''
        Return the best feats for train, val, and test when the top feature indices is known
        '''
        best_feats_train = self.feats_train[:,self.top_feat_indices]
        best_feats_val = self.feats_val[:,self.top_feat_indices]
        best_feats_test = self.feats_test[:,self.top_feat_indices]
        
        return (best_feats_train, best_feats_val, best_feats_test)
    
    def normalize_feats(self, norm = 'min-max', data_type = np.float32):
        '''
        Normalizes features according to input norm
        @param norm: Values 'l1'/'l2'/min-max/standardize/mean
                     If 'min-max', scales the values between 0 and 1. 
                     If 'standardize', standardizes the values as a Gaussian distribution with mean 0 and std 1
                     If 'mean': Reduces mean of distribution to zero.
        '''
        print("Normalizing data")
        if norm == 'l1' or norm == 'l2':
            transformer = Normalizer(norm = norm, copy = False)  
        elif norm == 'min-max':
            transformer = MinMaxScaler(copy = False)
        elif norm == 'standardize':
            transformer = StandardScaler(with_mean= True, with_std=True, copy = False)
        elif norm == 'mean':
            transformer = StandardScaler(with_mean= True, with_std=False, copy = False)
        
        if issparse(self.feats_train) and norm != 'l1' and norm != 'l2':
            self.feats_train = csr_matrix(transformer.fit_transform(self.feats_train.todense()), dtype = data_type )
            self.feats_val = csr_matrix(transformer.transform(self.feats_val.todense()), dtype = data_type )
            self.feats_test = csr_matrix(transformer.transform(self.feats_test.todense()), dtype = data_type )
        else:
            self.feats_train = transformer.fit_transform(self.feats_train)
            self.feats_val = transformer.fit_transform(self.feats_val)
            self.feats_test = transformer.fit_transform(self.feats_test)