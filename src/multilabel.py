# -*- coding: utf-8 -*-

# from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_recall_fscore_support, label_ranking_loss,  jaccard_similarity_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2

import numpy as np
from scipy.sparse import csc_matrix

def fit_ml(x_train, y_train, label_vocab, seed, c_type = 'RF', p_type = 'binary_relevance', best = False):
    '''
    Train classifiers for a multilabel algorithm, using different multilabel problem transformation strategies.
    @param x_train: training data
    @param y_train: training labels
    @param label_vocab: dictionary of all labels in the vocab (unsorted)
    @param seed: seed for randomization if required
    @param c_type: (RF/NB) the type of ML c_type to use.
    @param p_type: (binary_relevance): The multilabel problem transformation technique
    @param best: True to use the selected best features
    @return classifiers: list of trained models, such that the index in the list corresponds to the column index for the classes in the data
    '''
    classifiers = []
    if p_type == 'binary_relevance':
        for cur_label in range(len(label_vocab)):
            if c_type == 'RF':
                cur_classifier = RandomForestClassifier(n_estimators=100, random_state=seed, criterion='entropy')
            elif c_type == 'NB':
                cur_classifier = MultinomialNB()
            if best:
                classifiers.append(cur_classifier.fit(x_train[cur_label][0], y_train.getcol(cur_label).toarray()))
            else:
                classifiers.append(cur_classifier.fit(x_train, y_train.getcol(cur_label).toarray()))
    return classifiers

def get_pred_ml(classifiers, x_test, test_type = 1, best = False):
    '''
    Get the prediction for the test_type set, given the trained models for a multilabel algorithm.
    @param classifiers: list of trained classifiers, one for each label. Indices of classifiers in the list correspond to the label column indices
    @param test_type = 1/2: 1 for validation, 2 for test_type data
    @param best = True if best features need to be selected.
    @return scipy sparse CSC matrix of predicted labels
    '''
    if best:
        preds = np.zeros(shape=(x_test[0][test_type].shape[0],len(classifiers)))
    else:
        preds = np.zeros(shape=(x_test.shape[0],len(classifiers)))

    for i, cur_classifier in enumerate(classifiers):
        if best:
            preds[:,i] = cur_classifier.predict(x_test[i][test_type])
        else:
            preds[:,i] = cur_classifier.predict(x_test)
    
    return csc_matrix(preds)
        
def get_pred_prob_ml(classifiers, x_test, test_type = 1, best = False):
    '''
    Get probability of predictions for each label and instance
    @param classifiers: list of trained classifers, one per label, corresponding to the label indices in the test set
    @param test_type = 1/2: 1 for validation, 2 for test_type data
    @param best = True if best features need to be selected.
    @parma x_test: test set
    @return array of prediction probabilities, shape(n_inst, n_labels)
    '''
    if best:
        pred_probs = np.zeros(shape=(x_test[0][test_type].shape[0],len(classifiers)))
    else:
        pred_probs = np.zeros(shape=(x_test.shape[0],len(classifiers)))
    
    for i, cur_classifier in enumerate(classifiers):
        try:
            pred_probs[:,i] = cur_classifier.predict_proba(x_test)[:,1]
        except:
            pred_probs[:,i] = 1.0 - cur_classifier.predict_proba(x_test)[:,0] #when there are no instances for a class, RF returns a 1D array with prob of absent only.
#             print("Label ", i)
#             print(cur_classifier.predict_proba(x_test))
    return pred_probs
                
def get_class_from_probs(y_prob, th = 0.5):
    '''
    Returns scores for the obtained predictions
    @param y_preds: numpy array of shape (num_of_inst, num_of_labels), where each cell contains the prob of that label being present. 
    @param th: threshold for assigning a label as a true class
    @return scipy sparse csc matrix of most probable class
    '''
    return csc_matrix((y_prob >= th).astype(int))
 
                
# def get_ml_classifier(classifier, seed, ml_type =  'binary_relevance'):
#     '''
#     Create a multilabel setup of a base classifier 
#     @param classifier (NB/RF/SVM): the base classifier to use
#     @param ml_type (binary_relevance/classifier_chains): conversion technique to use to make base classifier compatible with multilabel data
#     @return trained model
#     '''
#     if ml_type == 'binary_relevance':
#         if classifier == 'NB':
#             classifier = BinaryRelevance(MultinomialNB(), require_dense = [False, True])
#         elif classifier == 'RF':
#             classifier = BinaryRelevance(RandomForestClassifier(n_estimators=100, random_state=seed), require_dense = [False, True])
#         elif classifier == 'SVC':
#             classifier = BinaryRelevance(SVC(), require_dense = [False, True])
#     elif ml_type == 'classifier_chain':
#         if classifier == 'NB':
#             classifier = ClassifierChain(MultinomialNB(), require_dense = [False, True])
#         elif classifier == 'RF':
#             classifier = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=seed), require_dense = [False, True])
#     
#     return classifier

# def train(classifier, x_train, y_train):
#     '''
#     Trains a classifier and returns the trained model
#     @param x_train: feature set 
#     @param y_train: label set. 2D array, such that rows represent instances, and column, different labels. Binary entries in the array - for present or absent. 
#     @param classifier: the classifier to train the model using
#     @return trained classifier
#     '''
#     classifier.fit(x_train, y_train)
#     return classifier
#  
#  
# def test(classifier, x_test):
#     '''
#     Returns the label predictions for unknown samples
#     @param classifier: trained model
#     @x_test: test feature set
#     @return label predictions for x_test given trained classifier (scipy sparse matrix, binary for labels present/absent)
#     '''
#     return classifier.predict(x_test)

def score(y_true, y_pred, y_score, num_labels = None, metrics = ['label_f_score']):
    '''
    Scores the predicted labels with respect to true labels in a multilabel, multiclass setup
    @param y_true: gold standard labels
    @param y_pred: predicted labels
    @param y_score: confidence of predictions to calculate ranking loss
    @param metrics: evaluation metrics to score on
    @return dictionary of {metric:score} for all metrics in question 
    '''
    scores = dict()
    
    if 'hamming_loss' in metrics:
        scores['hamming_loss'] =  hamming_loss(y_true = y_true, y_pred = y_pred)
    
    if 'subset_accuracy' in metrics:
        scores['subset_accuracy'] =  accuracy_score(y_true = y_true, y_pred = y_pred)
        
    if 'sample_f_score' in metrics:
        scores['sample_prec'], scores['sample_recall'], scores['sample_f_score'], _ = precision_recall_fscore_support(y_true = y_true, y_pred = y_pred, average='samples')

    if 'accuracy' in metrics:
        scores['accuracy'] =  jaccard_similarity_score(y_true = y_true, y_pred = y_pred)
        
    if 'ranking_loss' in metrics:
        scores['ranking_loss'] = label_ranking_loss(y_true = y_true, y_score= y_score)
    
    if 'micro_f_score' in metrics:
        scores['micro_prec'], scores['micro_recall'], scores['micro_f_score'], _ = precision_recall_fscore_support(y_true = y_true, y_pred = y_pred, average='micro')
        
    if 'macro_f_score' in metrics:
        scores['macro_prec'], scores['macro_recall'], scores['macro_f_score'], _ = precision_recall_fscore_support(y_true = y_true, y_pred = y_pred, average='macro')
       
    if 'label_f_score' in metrics:
        scores['label_prec'], scores['label_recall'], scores['label_f_score'], _ = precision_recall_fscore_support(y_true = y_true, y_pred = y_pred, average= None, labels = [i for i in range(num_labels)])
    return scores
        
def select_best_feats(x_train, y_train, x_val, x_test, label_vocab, k = 100, ntype = 'fixed', metric = 'chi2'):
    '''
    Selects the top k (number or percentile) features based on the metric provided, and returns the new feature matrices for training, validation and testing features
    @param x_train: training feature set
    @param y_train: training labels
    @param x_val: Validation features
    @param x_test: training features
    @param label_vocab: dictionary with vocabulary of all possible labels
    @param k: number or percentile of features to select
    @param ntype (fixed/percentile): What does k represent, number of features ot select, or the percentile of features to select
    @return best training, val and test feat matrices
    '''
    if ntype == 'fixed':
        selector = SelectKBest(chi2, k = k)
    elif ntype == 'percentile':
        selector = SelectPercentile(chi2, percentile=k)
    
    best_feats = dict()
    
    for cur_label in range(len(label_vocab)):
        best_x_train = selector.fit_transform(x_train, y_train.getcol(cur_label).toarray())
        best_x_val = selector.transform(x_val)
        best_x_test = selector.transform(x_test)
        best_feats[cur_label] = (best_x_train, best_x_val, best_x_test)
    
    return best_feats 


def calibrate_th(y_pred_probs, y_true):
    '''
    Return the threshold for which max micro-averaged F-score is obtained.
    Calculates micro-averaged precision, recall and f-score at different thresholds for classification, in the range [0.25, 0.75], in the steps of 0.05
    @param y_pred_probs: 2d array with prediction prob of each label for each instance
    @param y_true: the gold standard labels
    '''
    optimal_th = 0.0
    max_f_score = 0.0
    for th in np.arange(0.0,1.05,0.05):
        y_pred = get_class_from_probs(y_pred_probs, th = th)
        prec, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average = 'micro')
        if f_score > max_f_score:
            max_f_score = f_score
            optimal_th = th
        print("Threshold: ", th, " Precision: ", prec, " Recall: ", recall, " F-score: ", f_score )
    
    print("Optimal threshold is: ", optimal_th)
    return optimal_th