import utils
from config import Config
from model_utils import plot_multiple_roc

cfg = Config('../config/config.xml')
task = 'in_hosp'
task_title = 'In hospital mortality'

score1 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_bow.pkl.gz', pickle_data = True, compress = True)
score2 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_sdae.pkl.gz', pickle_data = True, compress = True)
score3 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_doc2vec.pkl.gz', pickle_data = True, compress = True)
score4 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_ensemble.pkl.gz', pickle_data = True, compress = True)
score5 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_bocui.pkl.gz', pickle_data = True, compress = True)
score6 = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'test_pred_score_sdae_bocui.pkl.gz', pickle_data = True, compress = True)

y_true = [int(line.strip()) for line in open('../input/'+task+'_gs.txt')]

plot_multiple_roc(y_true, y_scores = [score1, score2, score3, score4, score5, score6], sys_labels = ['BoW', 'SDAE-BoW', 'Doc2vec', 'SDAE-BoW+doc2vec', 'BoCUI', 'SDAE-BoCUI'], task = task_title)
