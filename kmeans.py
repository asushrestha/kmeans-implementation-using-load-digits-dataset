import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)

y = digits.target

# k = len(np.unique(y))
k = 10  # for general purpose use k = constant
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    # prints score based on different metrics

clf = KMeans(n_clusters=k, init="random", n_init=10)
'''
n_clusters = k or 10 defines the number of clusters
init = "random" generates centroids randomly, visit sklearn website for other parameters
n_init = 10 defines the random number of generated for first iteration and picks the best centroid among 10    
'''
bench_k_means(clf,"1",data)