import numpy as np
import unittest
import multiprocessing as mp
from gensim.models import doc2vec as d2v

import nlp_utils

class Corpus(object):
    def __init__(self, string_tags=False):
        nlp_utils.tokenize_data(content_dir = '../input/test_data/', out_dir = '../input/test_data_tok/', tokenizer = 'ucto', lang = 'EN', skip_dup_token = True)
        self.string_tags = string_tags

    def _tag(self, i):
        return i if not self.string_tags else '_*%d' % i

    def __iter__(self):
        
        with open("../input/test_data_tok/lee_background.txt") as f:
            for i, line in enumerate(f):
                tokens = []
                for cur_token in line.split():
                    tokens.append(nlp_utils.preprocess_token(cur_token, lc = True, update_num = True, remove_punc = False, replace = False))
                yield d2v.TaggedDocument(tokens, [self._tag(i)])


class Doc2Vec_Test(unittest.TestCase):
    def __init__(self):
          
        super(Doc2Vec_Test, self).__init__()
          
        n_cores = mp.cpu_count()
        self.model = d2v.Doc2Vec(size=100, min_count=5, iter=10, workers=n_cores, seed = 1337)
#         self.assertEqual(self.model.docvecs.doctag_syn0.shape, (300, 100))
#         self.model = d2v.Doc2Vec(size = 400, dm = 0, min_count = 5, hs=0, negative=10, workers = n_cores, iter = 10, seed = 1337)

        corpus = Corpus()
        
        self.model.build_vocab(corpus)
        self.model.train(corpus)
        
        self.test_persistence(out_dir = "../output/doc2vec_test/")
        self.test_training()

    def test_persistence(self, out_dir):
        """Test storing/loading the entire model."""
        self.model.save(out_dir+'doc2vec_model')
        self.models_equal(self.model, d2v.Doc2Vec.load(out_dir+'doc2vec_model'))
    
    def test_similarity_unseen_docs(self):
        """Test similarity of out of training sentences"""
        rome_str = ['rome', 'italy']
        car_str = ['car']

        self.assertTrue(self.model.docvecs.similarity_unseen_docs(self.model, rome_str, rome_str) > self.model.docvecs.similarity_unseen_docs(self.model, rome_str, car_str))
    
    def test_training(self):
        """Test doc2vec training."""

        #self.models_equal(self.cur_doc2vec.model, self.cur_doc2vec.model2)
        self.model_sanity()
        
    def model_sanity(self):
        """Any non-trivial model on DocsLeeCorpus can pass these sanity checks"""
        fire1 = 0  # doc 0 sydney fires
        fire2 = 2  # doc 0 sydney fires
        tennis1 = 1924  # doc 1924 tennis

        # inferred vector should be top10 close to bulk-trained one
        
        with open("../input/test_data_tok/lee_background.txt") as f:
            for i, line in enumerate(f):
#                 doc0_tokens = line.split()
                doc0_tokens = []
                for cur_token in line.split():
                    doc0_tokens.append(nlp_utils.preprocess_token(cur_token, lc = True, update_num = True, remove_punc = False, replace = False))
                if i == 0:
                    break
        
#         print(doc0_tokens)
        doc0_inferred = self.model.infer_vector(doc0_tokens)
        sims_to_infer = self.model.docvecs.most_similar([doc0_inferred], topn=len(self.model.docvecs))
        f_rank = [docid for docid, sim in sims_to_infer].index(fire1)
        self.assertLess(f_rank, 10)
        
        # fire2 should be top30 close to fire1
        sims = self.model.docvecs.most_similar(fire1, topn=len(self.model.docvecs))
        f2_rank = [docid for docid, sim in sims].index(fire2)
        self.assertLess(f2_rank, 30)

        # same sims should appear in lookup by vec as by index
        doc0_vec = self.model.docvecs[fire1]
        sims2 = self.model.docvecs.most_similar(positive=[doc0_vec], topn=10)
        sims2 = [(id, sim) for id, sim in sims2 if id != fire1]  # ignore the doc itself
        sims = sims[:9]
        self.assertEqual(list(zip(*sims))[0], list(zip(*sims2))[0])  # same doc ids
        self.assertTrue(np.allclose(list(zip(*sims))[1], list(zip(*sims2))[1]))  # close-enough dists

        # sim results should be in clip range if given
#         clip_sims = self.model.docvecs.most_similar(fire1, clip_start=len(self.model.docvecs) // 2, clip_end=len(self.model.docvecs) * 2 // 3)
#         sims_doc_id = [docid for docid, sim in clip_sims]
#         for s_id in sims_doc_id:
#             self.assertTrue(len(self.model.docvecs) // 2 <= s_id <= len(self.model.docvecs) * 2 // 3)

        #alien doc should be out-of-place among fire news
        self.assertEqual(self.model.docvecs.doesnt_match([fire1, tennis1, fire2]), tennis1)

        # fire docs should be closer than fire-tennis
        self.assertTrue(self.model.docvecs.similarity(fire1, fire2) > self.model.docvecs.similarity(fire1, tennis1))
        
    def models_equal(self, model, model2):
        # check words/hidden-weights
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertTrue(np.allclose(model.syn0, model2.syn0))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        # check docvecs
        self.assertEqual(len(model.docvecs.doctags), len(model2.docvecs.doctags))
        self.assertEqual(len(model.docvecs.offset2doctag), len(model2.docvecs.offset2doctag))
        self.assertTrue(np.allclose(model.docvecs.doctag_syn0, model2.docvecs.doctag_syn0))


if __name__ == '__main__':
    doc2vec_test = Doc2Vec_Test()

