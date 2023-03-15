---
title: Clustering
date: 2023-01-02 00:00:00
description: Clustering
tags: 
 - Machine Learning
 - Clustering
---

## Business  Applications  
- E-commerce : Group similar customers based on there purchasing behaviour, money spending patterns, products used, geogrophical location etc.
- Image Segmentation : Grouping/ Clustering similar pixels
- Manual labeling  by clustering large data to smaller number of clusters and labeling just the clusters

## Dunn Index
- Small Intra Clustering Distance and Large Inter Clustering Distance
- For ${C_k}$ clusters with $d(i,j)$ as Inter-cluster distance between $C_i$ & $C_j$ , $d'(i)$ as Intra-cluster distance of cluser $C_i$, **Larger Dunn Index** means better clustering

$$Dunn \ Index = \frac{min_{1\le i \le j \le k} \ \  d(i,j)}{max_{1 \le i \le k} \ d'(i)}$$
## Density Based Clustering 

- Cluster points give rise to dense regions 
- Noise points give rise to sparse regions

### Density based spatial clustering of noise (DBSCAN)

- Density at point `p` is the number of points within a hypersphere of radius `Eps` around `p`
- Dense region is a hypersphere of radius `Eps` that contains at least `Min Points` number of points
- Core point : If `p` has >= `Min Points` in an `Eps` radius around it 
- Border Point : if `p` is not a core point but `p` belongs to the neighbourhood  of `q` (core point) i.e `dist(p,q) <= Eps` and `q` is core
- Noise point : if `p` is neither Core or Border is noise point
- Density Edge: if `p` & `q` are core points and `dist(p,q) <= Eps` then edge `p-q` is a density edge
- Density connected points: If there exists a path formed by Density Edges to reach from a point `p` to `q` then these are Density connected points

#### DBSCAN Algorithm
 1. $\forall \ x_i \ \epsilon \ D$ label all points as core, border or noise point using range queries build on KD-Tree
 2. Remove all noise point from the data 
 3. For each core point not assigned to a cluster 
	 1. Create a new cluster with point `p`
	 2. Add all the points that are density connected to `p` into this cluster
 4. For each border point assign it to the neareast core point's cluster

#### Hyper parameters Tuning
1. `Min Points` ~~ 2\*dimensionality 
2. Choose larger value of `Min Points` if dataset has noisy points
3. `Eps` $\forall x_i$ calculate $d_i$  i.e. distance of $x_i$ from the `Min Points` th  nearest neighbour
4. Sort $d_i$  in increasing order and pick `Eps` as the $d_i$ with the elbow in the plot

###  Complexity
- Time Complexity : $O(nLog(n))$
- Space Complexity : $O(n)$

!!! note
	- [Advatages of DBSCAN](https://en.wikipedia.org/wiki/DBSCAN#Advantages)
	- [Disadvantage of DBSCAN](https://en.wikipedia.org/wiki/DBSCAN#Disadvantages)

## K-Means
K-Means gives us k sets of points from which k clusters are derived

### Optimisation Task : 
For $D_i = \{x_1, x_2, ... , x_n \}$ find $\{C_j\}_k$ and  there by $\{S_j\}_k$ 
Objective function : 

$$argmin_{\{ C_j \}_k}  \sum_{i=1}^k \sum_{x\epsilon S_i} ||x-C_i||^2$$


such that  \  \  $\forall_i \  x_i \  \epsilon \  S_j$    \  \  \  and  \  \  \   $\forall_{i,j} \  S_i \cap  S_j = \phi$
where $C_i$ is the Centroid of the ith cluster of points $S_i$ and is given by 

$$C_i = \frac{1}{|S_i|} \sum_{x_j \ \epsilon \  S_i} x_j$$

#### Lloyd's Algorithm
Lloyd's algorithm is used to solve the above optimisation approximately as it is very hard to optimise accurately

1. Initialization : Randomly pick k points from $D$ and call them $\{C_j\}_k$ i.e. $C_1, C_2, ..., C_k$
2. Assignent : For each $x_i$ in $D$ select the neareast $C_j$  by calculating the $dist(x_i, C_j) \ \ \forall \ j \epsilon [1,k]$  and add $x_i$ to $S_j$ 
3. Recompute/Update Centroid : recalculate centroids using : $C_i = \frac{1}{|S_i|} \sum_{x_j \ \epsilon \  S_i} x_j$ 
4. Repeat Assignent and Update until the distance/change between old and new centroid is negligible and we can say that the algorithm has converged

#### K-mean++

- **Initialization Sensitivity** : Final clusters depend on the initialisation done in the first step
- Repeat K means with different initializations and pick the best clustering based on smaller intra-cluster and larger intra cluster distance
- Smart Initialization : **k-mean ++**
	- Pick the first centroid $C_1$ randomly from $D$
	- Create a distribution  $\forall \ x_i \ \epsilon \ D$ given by $d_i = dist(x_i, nearest \ \ Centroid)^2$
	- Pick a point from $D-C_1$ with probability proportional to $d_i$ and annotate it as $C_2$

### Limitations
 - Small number of outliers can affect the result of K-mean and K-means++ hugely
 - k-means has problens when clusters are of different sizes because k means tries to create clusters of similar sizes
- k-means has problens when clusters are of different densities because k means tries to create clusters of similar densities
- k-means has problens with Non-globular (Non-Convex) clusters
- Good hack to solve these limitations  is to increase k and then combining similar clusters

### K-Medoids
Instead of giving a mean value as centroid which may or maynot be in the dataset, we can give a already interpretable data point as the centroid. 

#### Partitioning around medoids (PAM)
1. Initialization : Same as K-means++
2. Assignment : Closest Medoid (Same as K-means)
3. Recompute / Update: Swap each medoid with a non medoid point and if the loss decreases keep the swap else undo the swap. If Swap is success than do the assignment again
	- $Loss = \sum_{j=1}^k \sum_{x_i \epsilon S_j} ||x_i - m_j||^2$
	- The distance metric can be replaced with a kernel, which is again a benefit over K-means

### Determing the right K
- Elbow-Method /Knee-Method :  Calculate Loss for different values of k and choose the k after which decrease in loss is insignificant (basically the elbow point)

$$Loss = \sum_{i=1}^k \sum_{x\epsilon S_i} ||x-C_i||^2$$
### Time Complexity
Training Time complexity : $O(nkdi)$

- n = number of points
- k = No of clusters
- d = dimensions of data
- i = no of iteration
- Typically k and i are small  so ~~ $O(nd)$

## Hierarchical Clustering 

### Agglomerative & Divisive Clustering
- Agglomerative clustering assumes all points are individual clusters and starts grouping these clusterns in each iteration based on some notion of similarity or distances
- Divisive Clustering is just the reverse process and it starts with one big cluster  and divides into smaller clusters
- Computaion Steps
	- Compute the proximity matrix 
	- Let each data point be a cluster 
	- Repeat 
		- Merge the two closest clusters 
		- Update the proximity matrix 
	- Until only a single cluster remains

### Inter Cluster Similarity or proximity
1. Single Link Agglomeration (Min Method) 

$$Sim(C_1,C_2) = min \ Sim(p_i,p_j) \ \ \forall \  p_i \epsilon C_1 ;\  p_j \epsilon C_2$$

2. Min breaks when data has outliers
3. Complete Link Agglomeration (Max Method) 

$$Sim(C_1,C_2) = max \ Sim(p_i,p_j) \ \ \forall \  p_i \epsilon C_1 ;\  p_j \epsilon C_2$$

4. Max breaks when data has large clusters or differnt size clusters or non circular clusters
5. Group Avg Method (Compromise between min and max) 

$$Sim(C_1,C_2) = \frac{\sum_{p_i \epsilon C_1, p_j \epsilon C_2} \ Sim(p_i,p_j)}{|C_1| * |C_2|}$$

6. All Min, Max and Avg methods can be kernelised
7. Centroid Method 

$$Sim(C_1,C_2) = Distance / Similarty \ between \ centroids$$

8. Wards method is same as Group Avg with similarity as distance squared

## Complexity
- Space comlexity : $O(n^2)$   
- Time complexity : $O(n3)$  to $O(n2Log(n))$
- Both are very high and increase with n which is why it is not used very much in big data


- [Really good clustering lecture](https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf)
- [Visualising Clusters](https://stats.stackexchange.com/questions/52625/visually-plotting-multi-dimensional-cluster-data)
- [How to choose the right clustering algorithm](https://towardsdatascience.com/clustering-101-how-to-choose-the-right-algorithm-for-your-application-fb1521ea13fc)
