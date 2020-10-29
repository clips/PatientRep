from utils import read_data
from config import Config

from scipy.stats import spearmanr, kendalltau ,pearsonr
from scipy.stats.mstats import normaltest
from scipy.stats.morestats import boxcox

def load_data(cfg, analysis, importance_type, task):
    """
    Loads datasets for vocabulary frequency and term error/importance score
    """
    train_vocab_freq = read_data(cfg.path_cfg.PATH_INPUT, 'train_corpus_freq_'+ cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '.json')
    
    if analysis == 'pca':
        term_importance = read_data(path_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'principal_comp_400.json') #@TODO: update dim
    elif analysis == 'sdae_mse':
        print(task)
        term_importance = read_data(path_dir =  cfg.path_cfg.PATH_OUTPUT, f_name = 'feat_importance_'+importance_type+'_'+task+'.json')
    
    #jSON file contains equivalent string values - hence, convert to float again
    for cur_term, imp in term_importance.items():
        term_importance[cur_term] = float(imp)
        
    return train_vocab_freq, term_importance

def calculate_correlation(train_vocab_freq, term_importance):
    """
    Calculates spearman and kendall correlations for vocabulary frequency and term importance score.
    The scores indicate whether there exists a relationship between frequency and importance or not. 
    It does not care about whether the relationship is monotonic (increasing at the same rate) or not.
    To check for monotonic relationship, consult pearson's correlation. However, in that case, check for normality or transform data to normal distribution first
    """
    x = list()
    y = list()
    
    for term, freq in train_vocab_freq.items():
        if term == 'OOV':
            continue
        x.append(freq)
        y.append(term_importance[term])
    
    spearman_corr, spearman_p = spearmanr(x, y)
    print("Spearman's correlation is: ", spearman_corr,
          "with probability score of: ", spearman_p  
          )
    
    kendall_corr, kendall_p = kendalltau(x, y)
    print("Kendall-tau correlation is: ", kendall_corr,
          "with probability score of: ", kendall_p  
          )
    
    print("Before calculating Persons correlation, checking if data is normally distributed.")
     
    print("Normality test for frequency: ", normaltest(x))
    print("Normality test for term importance: ", normaltest(y))
     
    print("Both - frequency, and term importance score are not normally distributed. Transforming them to normal distribution using box-cox transformation")
    
    x, _ = boxcox(x)
    y, _ = boxcox(y)
    
    pearson_corr, pearson_p = pearsonr(x, y)
    print("Pearson's correlation after boxcox transformation is: ", pearson_corr,
          "with probability score of: ", pearson_p  
          )

if __name__ == '__main__':
    cfg = Config('../config/config.xml')
    train_vocab_freq, term_imp = load_data(cfg, analysis = 'sdae_mse', importance_type='recon_error', task = 'pri_diag_cat')
    calculate_correlation(train_vocab_freq, term_imp)
