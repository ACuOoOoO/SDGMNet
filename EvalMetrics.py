"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.
Extracted from:
https://github.com/hanxf/matchnet/blob/master/eval_metrics.py
"""
import numpy as np
from scipy.io import loadmat
import operator
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
def ErrorRateAt95Recall(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    labels = np.squeeze(labels)
    scores = np.squeeze(scores)
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    FN = np.sum(labels[threshold_index:] == 1)

    return float(FP) / float(FP + TN)


def AP(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    labels = np.array(labels[np.argsort(distances)])
    labels_t = np.hstack((0, labels))
    hist = np.cumsum(labels_t)
    histdiff = np.diff(hist)
    p_r = 0
    threshold_indx = (np.array(np.where(histdiff != 0))).squeeze()
    for i in range(len(threshold_indx)):
        precision = np.sum(labels[0:threshold_indx[i]+1] == 1)/(threshold_indx[i]+1)
        p_r += precision
    return p_r/(np.sum(labels))

def dist_dist(labels, scores):
    t = np.multiply(labels, scores)
    matched_dst_vec = t[np.where(t != 0)]
    t = np.multiply((1-labels), scores)
    nonmatched_dst_vec = t[np.where(t !=0 )]
    x = np.linspace(0, 2, 1999)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.001)
    kde.fit(matched_dst_vec[:, None])
    matched_dens = kde.score_samples(x[:, None])
    matched_dens = np.exp(matched_dens)
    kde.fit(nonmatched_dst_vec[:, None])
    nonmatched_dens = kde.score_samples(x[:,None])
    nonmatched_dens = np.exp(nonmatched_dens)
    return matched_dens, nonmatched_dens, x

def ratio_inter_intra(des, labels):
    des_dict = dict()
    for idx, _class in enumerate(labels):
        if _class not in des_dict:
            des_dict[_class] = []
            des_dict[_class].append(des[idx])

    mu_table = np.array(list(map(lambda x: np.mean(des_dict[x], axis=1), des_dict)))
    R_intra = np.mean(np.linalg.norm(mu_table, axis=1))
    R_inter = np.linalg.norm(np.mean(mu_table, axis=0))
    rho = R_inter/R_intra
    return rho

if __name__=='__main__':
    data1 = loadmat('./data/eval/tr_libertyte_notredame.mat')
    data2 = loadmat('./data/eval/tr_libertyte_yosemite.mat')

    dist1 = np.array(data1['dist'])
    lbl1 = np.array(data1['lbl'])
    dist2 = np.array(data2['dist'])
    lbl2 = np.array(data2['lbl'])
    lbl = np.hstack([lbl1,lbl2])
    dist = np.hstack([dist1,dist2])
    FR,FN,FP = ErrorRateAt95Recall(lbl,1/(dist+1e-8))
    print(FN)
    print(FP)
    pd, nd, x = dist_dist(lbl, dist)

    plt.plot(x,pd)
    plt.plot(x,nd)
    plt.show()


