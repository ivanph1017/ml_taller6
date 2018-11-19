#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 00:47:19 2018

@author: ivan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.mixture import GaussianMixture as mix
from sklearn.cross_validation import StratifiedKFold


def main():
    data = pd.read_csv(filepath_or_buffer='seeds_dataset.csv')
    print('{}'.format(data.head()))
    data.columns = ['Area', 'Perimeter', 'Compactness', 'Kernel_Length', 
                     'Kernel_Width', 'Asymmetry_Coefficient', 
                     'Kernel_Groove_Length', 'Cluster']
    
    #obtener los datos segun estadistica descriptiva
    print('\nData description \n{}'.format(data.describe()))
    
    #extracts the last column and convert the rest of the df into an array
    target = data['Cluster'] - 1
    target_names = target.unique()
    
    X = data[['Area', 'Perimeter', 'Compactness', 'Kernel_Length', 
                     'Kernel_Width', 'Asymmetry_Coefficient', 
                     'Kernel_Groove_Length']]
    X = np.array(X).astype(float)
    
    #normalización de datos
    mms = MinMaxScaler()
    mms.fit(X)
    X_norm = mms.transform(X)
    print('\nNormalized data \n{}'.format(X_norm))
    #probability of cluster assignment
    random_state = np.random.RandomState(seed=10)
    kf=StratifiedKFold(target,n_folds=3, random_state=random_state)
    #from documentation All the folds have size trunc(n_samples / n_folds)
    train_ind,test_ind = next(iter(kf))
    X_train = X[train_ind]
    y_train = target[train_ind]
    #X_test
    #y_test
    
    gmm_calc(X_train,"full",y_train,X,target,target_names, random_state)
    """gmm_calc(X,"diag",a)
    gmm_calc(X,"spherical",a)
    gmm_calc(X,"tied",a)"""


def gmm_calc(X_train,cov,y_train,X,target,target_names, random_state):
    n_classes = len(np.unique(y_train))
    model=mix(n_components=n_classes,covariance_type=cov, 
              random_state=random_state)
    model.fit(X_train)
    index=1
    h = plt.subplot(2, n_classes / 2, index + 1)
    make_ellipses(model,h)
    
    for n, color in enumerate('rgb'):
        data = X[target == n]
        print('{}'.format(data))
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=target_names[n])
 
    plt.savefig('em_gmm_scatter_plot.png')
    
    y_predict=model.predict(X_train)
    print('\n{} \n{}'.format(cov, y_train))
    print('\n{} \n{}'.format(cov, y_predict))
    print('{}'.format(np.mean(y_predict==y_train)*100))
    #probs=model.predict_proba(X_train)
    #print probs[:3].round(10)
    #size=50*probs.max(1)**2
    #plt.scatter(X_train[:, 2], X_train[:, 3], c=labels, s=size, cmap='viridis')
    print('{}'.format([round(i,5) for i in  (metrics.homogeneity_score(y_predict, y_train),
           metrics.completeness_score(y_predict, y_train),
           metrics.v_measure_score(y_predict,y_train),
           metrics.adjusted_rand_score(y_predict, y_train),
           metrics.adjusted_mutual_info_score(y_predict,  y_train))]))

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm.covariances_[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

if __name__=="__main__":
    main()
