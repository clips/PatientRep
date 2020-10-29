import config as cfg
import string
import re
import ucto

def tokenize_molis_descr(dir_name, lexicon_in, lexicon_out):
    config = "tokconfig-nld"
    tokenizer = ucto.Tokenizer(config)
    
    pattern_num = re.compile("^[+-]?[0-9]*[.,]?[0-9]+$") #regex for numeric values
    pattern_single_letter = re.compile("^[A-Za-z]$")
    
    lexicon_token = set()
    out_file = open(dir_name+lexicon_out, 'w')
    for cur_lexicon in open(dir_name+lexicon_in, 'r'):
        n_tokens = tokenizer.process(cur_lexicon)
        for token in tokenizer:
            continue
        candidate = str(token)
        if not candidate in string.punctuation and not pattern_num.match(candidate): #and not pattern_single_letter.match(candidate):
            print(candidate)
            lexicon_token.add(candidate)
            
    for term in lexicon_token:
        out_file.write(term+'\n')
       
tokenize_molis_descr(dir_name='../'+cfg.PATH_RESOURCES, lexicon_in = 'molis_labs.txt', lexicon_out='molis_labs_last_token.txt')
          