# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
    
    k_means(X_train,"full",y_train,X,target,target_names, random_state)
    
def k_means(X_train,cov,y_train,X,target,target_names, random_state):
    #within clusters sum of squares
    sum_squared_dist = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X_train)
        sum_squared_dist.append(km.inertia_)
    
    plt.plot(K, sum_squared_dist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method')
    plt.savefig('k_means_elbow_method.png')
    plt.show()
    
    #valor de k=3 aproximadamente
    kmeans = KMeans(n_clusters=3, random_state=random_state).fit(X_train)
    centroids = kmeans.cluster_centers_
    print('\nCentroids: \n{}'.format(centroids))
    
    # Predicting the clusters
    n_classes = len(np.unique(y_train))
    index=1
    h = plt.subplot(2, n_classes / 2, index + 1)
    # make_ellipses(kmeans,h)
    
    for n, color in enumerate('rgb'):
        data = X[target == n]
        print('{}'.format(data))
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=target_names[n])
 
    plt.savefig('k_means_scatter_plot.png')
    
    y_predict = kmeans.predict(X_train)
    print('\nLabels \n{}'.format(y_predict))
    print('{}'.format(np.mean(y_predict==y_train)*100))
    print('{}'.format([round(i,5) for i in  (metrics.homogeneity_score(y_predict, y_train),
           metrics.completeness_score(y_predict, y_train),
           metrics.v_measure_score(y_predict,y_train),
           metrics.adjusted_rand_score(y_predict, y_train),
           metrics.adjusted_mutual_info_score(y_predict,  y_train))]))

#def make_ellipses(k_means, ax):
#    for n, color in enumerate('rgb'):
#        v, w = np.linalg.eigh(k_means.inertia_[n][:2, :2])
#        u = w[0] / np.linalg.norm(w[0])
#        angle = np.arctan2(u[1], u[0])
#        angle = 180 * angle / np.pi  # convert to degrees
#        v *= 9
#        ell = mpl.patches.Ellipse(k_means.means_[n, :2], v[0], v[1],
#                                  180 + angle, color=color)
#        ell.set_clip_box(ax.bbox)
#        ell.set_alpha(0.5)
#        ax.add_artist(ell)

if __name__=="__main__":
    main()
