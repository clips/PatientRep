import numpy as np
#np.random.seed(1337)

from collections import OrderedDict, defaultdict

from sklearn.decomposition import PCA as sklearn_pca
from sklearn.decomposition import IncrementalPCA as sklearn_inc_pca

#from matplotlib import pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from scipy.sparse.base import issparse

from utils import write_data

class PCA(object):
    def __init__(self, dir_out, feats, labels, n_classes):
        self._pca(dir_out, feats, labels, n_classes)
        
    def _pca(self, dir_out, feats, labels, n_classes):
        """
        Compute principal components of training feats for dimensionality reduction (unsupervised) using Principal Component Analysis algorithm
        Steps involved:
        1) Reduce the mean of the feature matrix to zero, and optionally standardize the feature values
        2) Compute covariance matrix for training features
        3) Perform eigenvector and eigenvalue decomposition, or optionally SVD to speed it up, of covariance matrix
        4) Sort eigenvectors according to descending order of eigenvalues
        5) Top k eigen vectors and eigen values gives 'k' principal components
        6)        
        """
        print("Computing Principal components")
#         feats.normalize_feats(norm = 'mean')
        feats.normalize_feats(norm = 'standardize')
        
#         print("Getting covariance matrix")
#         cov_mat = np.cov(feats.feats_train.T)
#         
#         print("Getting eigenvectors and eigenvalues")
#         eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#          
# #         print("Eigen vectors:", eig_vecs)
# #         print("Eigen values:", eig_vals)
#         
# #         np.linalg.svd(, full_matrices = True, compute_uv)
#         
#         for ev in eig_vecs:
#             np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#         print('Everything ok!')
#         
#         # Make a list of (eigenvalue, eigenvector) tuples
#         eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#         
#         # Sort the (eigenvalue, eigenvector) tuples from high to low
#         eig_pairs.sort(key=lambda x: x[0], reverse=True)
#         
#         # Visually confirm that the list is correctly sorted by decreasing eigenvalues
# #         print('Eigenvalues in descending order:')
# #         for i in eig_pairs:
# #             print(i[0])
#             
#         tot = sum(eig_vals)
#         var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
#         cum_var_exp = np.cumsum(var_exp)
# #         print("Explained variance:", cum_var_exp)
#         
#         matrix_w = np.hstack((eig_pairs[0][1].reshape(feats.feats_train.shape[1],1),
#                       eig_pairs[1][1].reshape(feats.feats_train.shape[1],1)))
# 
# #         print('Matrix W:\n', matrix_w)
#         
#         y_pca = feats.feats_train.dot(matrix_w)

        print("Performing Incremental PCA:")
        n_comp = 400
        if issparse(feats.feats_train):
            feats.feats_train = feats.feats_train.todense()
        pca = sklearn_inc_pca(n_components = n_comp, batch_size=500)
        y_pca = pca.fit_transform(feats.feats_train)
        
        print("Total explained variance ratio by given number of components:", np.sum(pca.explained_variance_ratio_))
#         print("Explained variance ratio: ", pca.explained_variance_ratio_)
        
        print("Components: ", pca.components_)
        
        rev_vocab = dict()
        for k, v in feats.feat_vocab_idx.items():
            rev_vocab[v] = k
        
        print("Features with maximum effect (absolute) across all components: ")
        total_effect = np.mean(np.square(pca.components_), axis = 0)
        
        pca_imp = OrderedDict()
        for cur_feat in np.argsort(-total_effect):
            pca_imp[rev_vocab[cur_feat]] = str(total_effect[cur_feat])
            
        write_data(pca_imp, out_dir = dir_out, f_name = 'principal_comp_'+str(n_comp)+'.json')
        
        #self.plot_pca(y_pca, labels, n_classes)

    def plot_pca(self, y_pca, labels, n_classes):
        label_inst = defaultdict(list)
        
        for i, cur_label in enumerate(labels):
            label_inst[cur_label].append(i)
        
        colors = self.get_cmap(n_classes)
        
        with plt.style.context('seaborn-whitegrid'):
            plt.figure()
            for cur_class in range(n_classes):
                
                plt.scatter(y_pca[label_inst[cur_class], 0],
                    y_pca[label_inst[cur_class], 1],
                    label=cur_class,
                    c=colors[cur_class],
                    alpha=0.03)
#             for i, cur_label in enumerate(labels):
#                 plt.scatter(y_pca[i, 0],
#                             y_pca[i, 1],
#                             c=colors(cur_label))
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(loc='right')
            plt.tight_layout()
            plt.show()
            plt.savefig('pca.png', format = 'png', dpi = 1000)
        

    def get_cmap(self, N):
        '''Returns a list of N distinct RGB colors.'''
        if N == 2:
            return ['red', 'blue']
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
        cols = [scalar_map.to_rgba(index) for index in range(N)]
        return cols
