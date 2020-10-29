# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse.base import issparse

import matplotlib as mt
mt.use('Agg')
from matplotlib import pyplot as plt

def get_train_val_test_inst(inst, val_size, test_size, shuffle = True, seed = 1337):
    '''
    Returns instances split for training, validation and test when number of instances in validation and test set are known
    @param inst: shuffled set of instance IDs in the dataset
    @param val_size: number of instances in validation set
    @param test_size: number of instances in test set
    '''
    
    assert (len(inst) - val_size - test_size) > (val_size+test_size), "Too small a dataset to split instances as required"
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(inst)
    
    train = inst[:-(val_size+test_size)]
    val = inst[-(val_size+test_size):-test_size]
    test = inst[-test_size:]
    return (train, val, test)

def get_stratified_train_val_test_split(y, seed = 1337):
    '''
    Return indices for training and testing by doing a stratified split to maintain class proportions in each split
    @param y: class labels for the complete dataset
    @param seed: random state seed for replicability
    @return indices of train, val and test set
    '''
    sss1 = StratifiedShuffleSplit(y, n_iter = 1, test_size = 0.2, random_state = seed)
    for train_idx, rest_idx in sss1: 
        
        y_new = [y[idx] for idx in rest_idx]
        sss2 = StratifiedShuffleSplit(y_new, n_iter = 1, test_size = 0.5, random_state = seed)
        
        for val_idx_new, test_idx_new in sss2:
            val_idx = [rest_idx[i] for i in val_idx_new]
            test_idx = [rest_idx[i] for i in test_idx_new]
    
#     print(len(train_idx), len(val_idx), len(test_idx))
    return(train_idx, val_idx, test_idx)

def resample(x_train, y_train, minority_prop = 0.3, algo = 'undersample', seed = 1337):
    """
    Resample training dataset instances so that minority samples are the given proportion in the complete dataset
    """   
    
    print("Total number of original instances:", x_train.shape[0])
    
    if algo == 'undersample':
        rus = RandomUnderSampler(ratio = minority_prop, random_state=seed)
        x_res, y_res = rus.fit_sample(x_train, y_train)
    
    print("Number of instances in resampled data: ", x_res.shape[0])
    return (x_res, y_res)

def score(y_true, y_pred, y_score, n_classes):
    """
    Calculate scores for predictions based on true labels
    @param y_true: gold standard labels
    @param y_pred: predicted labels
    @param y_score: prediction probabilities
    """
    assert y_pred is not None or y_score is not None, "Please input at least one of the two: class predictions, or class prediction probabilities"
    
    if y_pred is None:
        y_pred = np.zeros(y_score.shape[0], dtype = np.int32)
        for i in range(y_score.shape[0]):
            y_pred[i] = np.argmax(y_score[i])
    
    error_idx = _get_error_idx(y_true, y_pred)
          
    scores = dict()
    
    scores['prec_micro'], scores['recall_micro'], scores['f_score_micro'], _ = precision_recall_fscore_support(y_true, y_pred, average = 'micro', labels = np.arange(n_classes))
    scores['prec_macro'], scores['recall_macro'], scores['f_score_macro'], _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro', labels = np.arange(n_classes))
    scores['prec_weighted'], scores['recall_weighted'], scores['f_score_weighted'], _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted', labels = np.arange(n_classes))
    scores['prec_all'], scores['recall_all'], scores['f_score_all'], _ = precision_recall_fscore_support(y_true, y_pred, average = None, labels = np.arange(n_classes))
    
    if n_classes == 2:
        if y_score is not None:
            scores['auc'] = roc_auc_score(np.array(y_true), y_score[:,1])
            
#             fpr, tpr, thresholds = roc_curve(np.array(y_true), y_score[:,1], pos_label = 1) #using score of true class
#             scores['auc'] = auc(fpr, tpr)
#             
#             plot_roc([fpr],[tpr],[scores['auc']])
            
    conf_matrix = confusion_matrix(y_true, y_pred, labels = np.arange(n_classes))
    return scores, conf_matrix, error_idx

def plot_multiple_roc(y_true, y_scores, sys_labels, task = None):
    """
    Plot multiple ROC curves.
    @param y_true: true labels common to the curves
    @param y_scores: list of 2D scoring array for different systems
    @param sys_labels: Labels for different systems
    """
    fprs = list()
    tprs = list()
    auc_scores = list()
    
    for cur_y_score in y_scores:
        fpr, tpr, thresholds = roc_curve(np.array(y_true), cur_y_score[:,1], pos_label = 1) #using score of true class
        fprs.append(fpr)
        tprs.append(tpr)
        auc_scores.append(auc(fpr,tpr))
    
    plot_roc(fprs, tprs, auc_scores, sys_labels, task)    
    
def plot_roc(fprs,tprs,auc_scores, sys_labels = None, task = None):
    """
    Plot multiple roc curves
    @param fprs: list of false positive rates
    @param tprs: list of true positive rates
    @param auc_scores: list of AUROC scores
    @param sys_labels: labels for different systems
    @param task: Task for which we are plotting the curve
    """
    plt.figure()
    lw = 2 #linewidth
    linestyles = [':', '-', '-', '-', ':', '-'] #extend the list to plot more than 3 curves together
#     markers = ['v','^','o','s', '<', '>']
    colors = ['#E69F00', '#CC79A7', '#000000', '#F0E442', '#000000', '#0072B2']
    if sys_labels == None:
        label = 'ROC curve'
        
    for i in range(len(fprs)):
        label = sys_labels[i]
        
#         if not i:
#             plt.plot(fprs[i], tprs[i], color=colors[i],
#                          lw=lw, linestyle = linestyles[0], label=label+' (area = %0.2f)' % auc_scores[i])
#         else:
        plt.plot(fprs[i], tprs[i], color=colors[i],
                         lw=lw, linestyle = linestyles[i], label=label+' (area = %0.2f)' % auc_scores[i])
            
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    title = 'Receiver operating characteristic'
    if task is not None:
        title += ' for the task: '+task
    
    plt.title(title)
    
    plt.legend(loc="lower right")
    
    plt.savefig('../output/auc_roc.png')
    
def _get_error_idx(y_true, y_pred):
    """
    Returns indices of incorrectly predicted instances from the predictions array
    @param y_true: true label
    @param y_pred: predicted label
    """
    
    error_idx = set()
    
    for cur_idx, cur_y in enumerate(y_true):
        if cur_y != y_pred[cur_idx]:
            error_idx.add(cur_idx)
            
    return error_idx
    
def get_multilabel_stats(labels, labels_vocab):
    '''
    Get the number of instances present for each label (label frequency), and the average number of labels present per instance.
    @param labels: 2D numpy array ,shape (num_of_pt*num_of_labels), where an entry '1' represents that it is a diagnostic category for the last stay of the patients.
    @param labels_vocab: dictionary with vocabulary of different diagnostic categories with their corresponding IDs {code:id}.
    '''
    
    #label_cardinality: average num of labels per example
    #label_density: label_cardinality normalized by total num of labels
    #label_diversity: num of distinct label sets in the dataset (can be normalized by num of instances to indicate proportion of distinct label sets)
    
    label_sets = set()
    total_labels = 0
    
    for cur_pt in range(labels.shape[0]):
        total_labels += sum(labels[cur_pt])
        cur_label_set = set()
        for cur_label in range(labels.shape[1]):
            if labels[cur_pt][cur_label] == 1:
                cur_label_set.add(cur_label)
        label_sets.add(tuple(cur_label_set))
        
        
    label_cardinality = total_labels/labels.shape[0]
    label_density = label_cardinality/len(labels_vocab)
    label_diversity = len(label_sets)
    norm_label_diversity = label_diversity/labels.shape[0]
    
    print("Total num of individual labels in vocab:", len(labels_vocab))
#     print("Distinct label combinations present")
#     for cur_label in label_sets:
#         print(cur_label)
    
    print("Label cardinality (Average num of labels per example): ", label_cardinality)
    print("Label density (label_cardinality normalized by total num of labels): ", label_density)
    print("Label diversity (Num of distinct label sets in the dataset): ", label_diversity)
    print("Normalized label diversity (Proportion of distinct label sets in the dataset): ", norm_label_diversity)
    
def get_label_stats_sparse(labels, labels_vocab):
    '''
    For each label, print the number of instances of the label in the dataset
    @param labels: Scipy sparse (CSC) matrix for labels for each instance in a given set, typically train set.
    @param labels_vocab: dictionary with vocabulary of different labels (text) with their corresponding IDs {code:id}.
    '''
    
    for cur_label, cur_label_id in labels_vocab.items():
#         print(labels.getcol(cur_label_id))
#         print(labels.getcol(cur_label_id).nonzero())
        print(cur_label, " instances: ", len(labels.getcol(cur_label_id).nonzero()[0])) #subscripting with 0 to find out num of non zero rows
    
def get_label_stats_dense(y_true, labels, label_vocab = None):
    """
    Get number of instances for each class
    @param y_true: True class labels for each instance (not one hot)
    @param labels: all class labels in vocabulary 
    """
    label_freq = np.zeros(labels.shape[0], dtype = np.int32) 
    
    for cur_class in y_true:
        label_freq[cur_class] += 1
        
    for i in range(len(labels)):
        if label_vocab:
            print(label_vocab[i], " instances: ", label_freq[i]/len(y_true)*100)
        else:
            print("Class ", i, " instances: ", label_freq[i]/len(y_true)*100)