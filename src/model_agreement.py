import numpy as np
np.random.seed(1337)

from sklearn.metrics import cohen_kappa_score

from utils import read_data

def get_pred(pred_score):
    y_pred = np.zeros(pred_score.shape[0], dtype = np.int32)
    for i in range(pred_score.shape[0]):
        y_pred[i] = np.argmax(pred_score[i])
    return y_pred

pred_score_d2v = read_data('../output/', f_name = 'val_pred_score_d2v.pkl.gz', pickle_data = True, compress = True)
y_pred_d2v = get_pred(pred_score_d2v)

pred_score_sdae = read_data('../output/', f_name = 'val_pred_score_sdae.pkl.gz', pickle_data = True, compress = True)
y_pred_sdae = get_pred(pred_score_sdae)

pred_score_bow = read_data('../output/', f_name = 'val_pred_score_bow.pkl.gz', pickle_data = True, compress = True)
y_pred_bow = get_pred(pred_score_bow)

print("Kappa score for agreement between doc2vec and sdae: ", cohen_kappa_score(y_pred_d2v, y_pred_sdae))
print("Kappa score for agreement between sdae and bow: ", cohen_kappa_score(y_pred_sdae, y_pred_bow))
print("Kappa score for agreement between doc2vec and bow: ", cohen_kappa_score(y_pred_d2v, y_pred_bow))

