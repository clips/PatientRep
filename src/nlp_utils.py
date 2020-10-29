# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

import numpy as np
import math
# import matplotlib.pyplot as plt
import string
from collections import defaultdict, OrderedDict
import operator
import re

#import ucto
import os
import sys

def get_feats_bow_sklearn(content, remove_sw, sw_list = None, lang = 'EN', normalize = 'l1'):
    '''
    Returns bow representation of content, based on sklearn tokenizer and TF-IDF vectorizer
    @param content: list of strings. Each entry in the list is a document.
    @param remove_sw: if stopwords should be removed
    @param sw_list: a list of stopwords, if using a language other than English
    @param lang: The language to use.
    @param normalize: l1/l2/None. If None, the BOW representation will not be normalized.
    @return bag-of-words representation of given input data.
    
    PS the vectorizer removes all single letter feats. Hence, words like 'I', 'a', etc are removed.
    @todo: replace the tokenizer (or input tokenization pattern) to include single letter feats
    '''
    if remove_sw:
        if lang == 'EN':
            vectorizer = TfidfVectorizer(analyzer='word', lowercase = True, strip_accents='unicode', use_idf=False, norm = normalize, stop_words='english')
        elif not sw_list:
            print('Please input a compatible stopwords list, processing without removing stopwords.')
            vectorizer = TfidfVectorizer(analyzer='word', lowercase = True, strip_accents='unicode', use_idf=False, norm = normalize)
        else:
            vectorizer = TfidfVectorizer(analyzer='word', lowercase = True, strip_accents='unicode', use_idf=False, norm = normalize, stop_words=sw_list)
    else:
        vectorizer = TfidfVectorizer(analyzer='word', lowercase = True, strip_accents='unicode', use_idf=False, norm = normalize)
          
    feats = vectorizer.fit_transform(content).toarray()
    
#     print(vectorizer.get_feature_names())
    return feats 

def tokenize_pt_data(pids, content_dir, out_dir, tokenizer = 'ucto', lang = 'NL', skip_dup_token = True, folia_output = False):
    '''
    Tokenizes content and saves the tokenized data as a separate file
    @param pids: patient IDs
    @param content_dir: the directory which contains text files of documents for each patient
    @param out_dir: directory where tokenized documents should be saved
    @param tokenizer: which tokenizer to use
    @param lang: (NL/EN) the language of the text.
    @param skip_dup_token: True to skip tokenizing file that is already tokenized
    @param folia_output: True to write output in folia format.
    '''
    if lang == 'NL':
        config = "tokconfig-nld"
        
    elif lang == 'EN':
        config = "tokconfig-eng"
    
    if tokenizer == 'ucto':
        cur_tokenizer = ucto.Tokenizer(config, sentenceperlineinput = False, sentenceperlineoutput = True, sentencedetection = True, foliaoutput=False)
    
    tot_patients = len(pids)

    for i, cur_pid in enumerate(pids):
    
        if not i%100:
            print("Tokenizing ", i," of ", tot_patients, " patient data ")
#         tokenizer.process(cur_pt_content)
        if not skip_dup_token or not os.path.isfile(out_dir+str(cur_pid)+'.txt'):
            cur_tokenizer.tokenize(content_dir+str(cur_pid)+'.txt', out_dir+str(cur_pid)+'.txt')
        
def tokenize_data(content_dir, out_dir, tokenizer = 'ucto', lang = 'NL', skip_dup_token = True, folia_output = False):
    '''
    Tokenizes files in a given directory and saves the tokenized data as a separate file
    @param content_dir: the directory which contains text files to be tokenized
    @param out_dir: directory where tokenized documents should be saved
    @param tokenizer: which tokenizer to use
    @param lang: (NL/EN) the language of the text.
    @param skip_dup_token: True to skip tokenizing file that is already tokenized
    @param folia_output: True to write output in folia format.
    '''
    if lang == 'NL':
        config = "tokconfig-nld"
        
    elif lang == 'EN':
        config = "tokconfig-eng"
    
    if tokenizer == 'ucto':
        cur_tokenizer = ucto.Tokenizer(config, sentenceperlineinput = False, sentenceperlineoutput = True, sentencedetection = True, foliaoutput=False)
        
    for i, cur_file in enumerate(os.listdir(content_dir)):
        
        if not i%100:
            print("Tokenizing document", i)
#         tokenizer.process(cur_pt_content)
        if not skip_dup_token or not os.path.isfile(out_dir+cur_file):
            cur_tokenizer.tokenize(content_dir+cur_file, out_dir+cur_file)
        
def preprocess_token(token, lc, update_num, remove_punc, replace = True, fix_labs = False, labs = None):
    '''
    Pre-process tokens to replace numbers with 'numeric_val', to lower case tokens if required, and to replace a space in between two words in a token with '_'
    Removes puntuation(s), and replaces time and measurement values in format 20/30/60 with 'time_meas'.
    @param token: the token to process
    @param lc: True to lowercase
    @param update_num: True to replace numbers with 'numeric_val'
    @param remove_punc: True to remove punctuation
    @param fix_labs: (for NL) True to identify lab test from incorrectly concatenated tokens
    @param labs: lab test names
    @param replace: True to replace numbers and time/measurement tokens with equivalents. False to remove these tokens
    @return the new token
    '''
    
    if fix_labs:
        lab = _get_test_name(token, labs)
        if lab:
            token = lab
    
    if lc:
        token = token.lower()
    
    #remove sequences of punctuation
    if remove_punc:
        token = token.strip(string.punctuation)
    
    #pattern date, time, BP etc
    token = _update_date_time_meas(token, replace)

    #numbers
    token = _update_numbers(token, update_num, replace)

    token = token.replace(" ","_")        
    return token

def _update_date_time_meas(token, replace):
    pattern_excl = re.compile("^([0-9]+[/:.,]+[0-9]*)+['a.m.','a.m', 'am.', 'am' ,'pm','p.m.', 'p.m', 'pm.']*$")
    
    if pattern_excl.match(token):
        if replace:
            token = 'time_meas'
        else:
            token = ''
            
    return token

def _update_numbers(token, update_num, replace):
    pattern_num = re.compile("^[+-]?[0-9]*[.,]?[0-9]+$") #regex for numeric values
    if update_num and pattern_num.match(token):
#         print(token)
        if replace:
            token = "numeric_val"
        else:
            token = ''
            
    return token

def _get_test_name(token,labs):
    '''
    Fix the lab tests and corresponding results which have been concatenated due to incorrect conversion of tables from RTF.
    @param token: the token which needs to be matched and corrected if it contains a lab test name
    @param molis_lexicon: MOLIS lexicon of all lab test names
    @return list containing different test names present within the token
    '''     
    regex_concat = '[^A-Za-z]*[A-Za-z]+[-+]*[A-Za-z]*[0-9]+'
    pattern = re.compile(regex_concat)
    
    if pattern.match(token):
#         print(token)
        for cur_lab in labs.keys():
            if token.lower().startswith(cur_lab):
#                 print('Current token', token.lower())
#                 print('matched lab', cur_lab)
                return cur_lab
#         print(token)
    return None
    
def remove_lf_terms(vocab, freq_th = 5):
    '''
    Given a vocab, remove the terms with corpus frequency lower than of equal to a given threshold
    @param vocab: dictionary {token: [corpus freq, set of pt that the token occurs for]} ... contains corpus freq and DF counts
    @param freq_th: the threshold for a term to occur in the vocabulary
    @return vocab (in the same data structure) after removing terms with low freq in corpus
    '''
    print("Removing terms with corpus freq < ", freq_th)
    tot_terms = len(vocab)
    lf_terms = 0
    
    f = open('lf_'+str(freq_th)+'.txt', 'w')
    
    for term, tf in list(vocab.items()):
        
        if term == 'OOV':
            continue
        
        if tf[0] < freq_th:
            
            val = vocab[term]
            vocab['OOV'][0] += val[0]
            vocab['OOV'][1] = vocab['OOV'][1].union(val[1])
            
            f.write(term+'\n')
            del vocab[term]
            lf_terms += 1
    f.close()
    
    print("Removed ", lf_terms, " of total ", tot_terms, " terms")
    print("New vocab size, ", len(vocab))
    return vocab

def get_df(vocab):
    '''
    Get a vocabulary of terms and their document frequency. Remove corpus frequency information.
    @param vocab: dictionary {token: [corpus freq, set of pt that the token occurs for]} ... contains corpus freq and DF counts
    @return vocab dictionary with its document freq counts, {term:df of term}
    '''
    for k, v in vocab.items():
        vocab[k] = len(v[1])
    print("Created dictionary {token:document frequency}")
    return vocab

def calculate_idf(vocab, ndocs):
    '''
    Returns logarithmic IDF scores for terms in vocab
    @param vocab: dictionary {term:document freq of term}
    @param ndocs: total num of docs in corpus
    @return log idf of vocab terms
    '''        
    v1 = math.log(ndocs)
#     print("log of total num of docs in the corpus", v1)
    for k, v in vocab.items():
        vocab[k] = v1 - math.log(v) #get log of IDF scores from document freq.
    
    print("Calculated IDF scores")
    return vocab

def get_tf_idf(term_freq, idf):
    '''
    Returns smoothed (+1) to TF-IDF value before taking log) log TF-IDF score for terms.
    @param term_freq: list of dictionaries: different instances have a different dictionary, ordered as items of list. Dictionary contains tokens and corresponding token freq in that instance.
    @param idf: dictionary containing tokens in training set vocabulary, and corresponding logarithmic IDF scores.
    @return: list of dictionaries, where dictionary contains TF IDF score of tokens in that dictionary.
    '''
    print("Calculating TF-IDF scores")
    for token_dict in term_freq:
        for term, tf in list(token_dict.items()):
            if term in idf:
                token_dict[term] = math.log( math.exp(math.log(tf) + idf[term]) +1) #sum because both are logarithmic values; IDF is already log value
            else:
                del token_dict[term] #deleting terms not present in vocab
    return term_freq

def get_tf(term_freq, idf):
    '''
    Returns smoothed (+1) TF score for tokens in the document.
    @param term_freq: list of dictionaries: different instances have a different dictionary, ordered as items of list. Dictionary contains tokens and corresponding token freq in that instance.
    @param idf: dictionary containing tokens in training set vocabulary, and corresponding logarithmic IDF scores.
    @return: list of dictionaries, where dictionary contains smoothed TF score of tokens in that dictionary.
    '''
    print("Calculating smoothed TF scores")
    for token_dict in term_freq:
        for term, tf in list(token_dict.items()):
            if term in idf:
                token_dict[term] = tf+1 #add 1 smoothing
            else:
                del token_dict[term] #deleting terms not present in vocab
    return term_freq