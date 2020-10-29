from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score

class Cluster(object):
    """
    Class with supporting functions for clustering a data, predicting cluster assignments, and thereby evaluating clustering quality
    """
    def __init__(self, data_in, data_test, labels_test, n_clusters, algo='k_means'):
        '''
        Initialization parameters for supported clustering algorithms
        @param n_clusters: number of clusters to generate
        @param algo: (k_means): algorithm to use for clustering
        '''
        self.n_clusters = n_clusters
        self.algo = algo
        
#         for n in range(50,1050,50):
#             self.n_clusters = n
        self._cluster(data_in)
        self._predict(data_test)
        print("Results for k = ", self.n_clusters)
        self._evaluate_cluster_supervised(labels_test, eval_type = 'ext')
        self._evaluate_cluster_unsupervised(data_test, cluster_labels = self.labels_pred)
        
    def _cluster(self, data_in):
        """
        Performs clustering; trains a clustering model on the given input data
        @param data_in: data to cluster
        """
        if self.algo == 'k_means':
            self.clusters = KMeans(n_clusters= self.n_clusters).fit(data_in)
            self.labels_pred = self.clusters.labels_
#             print(self.labels_pred)
    
    def _predict(self, data_in):
        """
        Predict cluster labels for given data
        """
        self.labels_pred = self.clusters.predict(data_in)
#         print(self.labels_pred)
               

    def _evaluate_cluster_supervised(self, labels_true, eval_type = 'ext'):
        '''
        @param labels_true: true labels of instances
        @param eval_type: (ext/int) extenal or instrinsic evaluation
        '''
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true, self.labels_pred)
        print("Homogeneity: ", homogeneity)
        print("Completeness: ", completeness)
        print("V-measure: ", v_measure)
        
        return (homogeneity, completeness, v_measure)
    
    def _evaluate_cluster_unsupervised(self, feats, cluster_labels):
        """
        @param feats: data that has been clustered
        @param cluster_labels: generated cluster assignments for each sample
        """
        
        silhouette_avg = silhouette_score(feats, cluster_labels)
        print("Average silhouette score for all samples is: ", silhouette_avg)

        return silhouette_avg