### Comparison between K-means and EM-GMM algorithms

In order to achieve consistent results, both models were executed with a seed value of 10.  

Beforehand, the descriptive analysis defined in 3 clusters is shown below

![Scatter plot](https://raw.githubusercontent.com/ivanph1017/ml_taller6/master/k_means_scatter_plot.png)

#### K-means algorithm with K-means++ initialization  

Predicted Labels   
[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2  
 2 2 2 0 0 0 0 0 2 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  
 0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2  
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 2]  

Accuracy level: 85.5072463768116  

Homogeneity score: 0.61607  
Completeness score: 0.61032  
V measure score: 0.61318  
Adjusted random score: 0.61854  
Adjusted mutual info score: 0.605   

The elbow method shows that the optimum k value is 3:  

![Elbow method](https://raw.githubusercontent.com/ivanph1017/ml_taller6/master/k_means_elbow_method.png)

#### Expectation Maximization - Gaussian Mixture Model   

Predicted Labels   
[0 0 2 2 0 0 0 0 0 0 0 0 0 1 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2  
 2 2 2 2 2 0 0 0 2 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  
 0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2]  

Accuracy level: 83.33333333333334    

Homogeneity score: 0.6071    
Completeness score: 0.59903    
V measure score: 0.60304    
Adjusted random score: 0.58562   
Adjusted mutual info score: 0.59356    

The elbow method shows that the optimum k value is 3:  

![EM-GMM scatter plot](https://raw.githubusercontent.com/ivanph1017/ml_taller6/master/em_gmm_scatter_plot.png)
