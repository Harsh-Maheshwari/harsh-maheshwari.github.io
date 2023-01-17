---
title: Support Vector Machines
date: 2022-12-31 00:00:00
description: Support Vector Machines
tags: 
 - Machine Learning
 - Support Vector Machines
---

## Geometric Intution

SVMs try to find a separating hyper plane $\pi$ which maximises the margin, where margin is the distance between the planes $\pi+$ and $\pi-$. 

- $\pi+$ is plane parralel to $\pi$ (separating plane) which passes through first positive point 
- $\pi-$ is plane parralel to $\pi$ (separating plane) which passes through first negative point
- Points on through which $\pi+$ and $\pi-$ pass through are called support vectors

### Convex Hull
Convex Hull is the smallest convex ploygon such that all the points are inside or on the polygon 

- Construct a convex hull for positive and negative points separately
- Find the shortest line connecting the two convex hulls
- The perpendicular bisector of the shortest line is the margin maximising hyperplane

## Decision Plane & Support Vectors 

$$\pi : w^Tx + b = 0$$

if support vectors are :

$$\pi+ : w^Tx + b = 1$$

$$\pi- : w^Tx + b = -1$$

### Constraint Optimisation 
#### Hard Margin SVM:

$$(w^*, b^*) = argmax(w,b) \frac{2}{||w||}$$

such that   

$$y_i(w^Tx_i+b) \ge 1 \ \  \forall  \ \   x_i$$

- All points should be correctly classified in the hard margin optimisation 
- $\frac{2}{||w||}$ is the margin distance between $\pi+$ and $\pi-$

#### Primal form - Soft Margin SVM

$$(w^*, b^*) = argmin(w,b) \frac{||w||}{2} + C\frac{1}{n} \sum_i^n \zeta_i$$

such that 


$$y_i(w^Tx_i+b) \ge 1 - \zeta_i \ \ \forall i$$

and  

$$\zeta_i \ge 0$$

- $0.5*||w||$ is the inverse of margin
- $\frac{1}{n} \sum_i^n \zeta_i$  is the average distance of missclassified points
- If the point is correctly classified : $\zeta_i = 0$ 
- The point is further away from the correct hyperplane  as $\zeta_i$ increases for a point  
- $C$ is multipled to the loss finction (Hinge Loss see ahead) hence as $C$ increases there is a tendency to overfit and as $C$ decreases there is a tendency to underfit.
- This is also called $C-soft-SVM$

#### Dual form - Soft Margin SVM

$$max_{\alpha} \  (\sum_i^n \alpha_i - 0.5\sum_i^n\sum_j^n\ \alpha_i\alpha_j\ y_iy_j\ x_i^Tx_j)$$

such that 

$$\alpha_i\ge0$$

$$\sum_i^n\alpha_iy_i = 0$$

To get the class label in this dual form we have this function

$$f(x_q) = \sum_i^n \alpha_i y_i\ x_i^Tx_q + b$$

!!! note 
	- $x_i^Tx_q$ is the cosine similarity if $x$ is normalised.
	- Theoritically we can replace  $x_i^Tx_q$ with any kind of similarity function : kernel function

## Kernel Trick
Using Kernel tricks SVM can handle non linearly separaed datasets by the way of implicit feature transformation

$$max_{\alpha} \  (\sum_i^n \alpha_i - 0.5\sum_i^n\sum_j^n\ \alpha_i\alpha_j\ y_iy_j\ K(x_i, x_j))
$$

such that 

$$\alpha_i\ge0$$

$$\sum_i^n\alpha_iy_i = 0$$

To get the class label in this dual form we have this function

$$f(x_q) = \sum_i^n \alpha_i y_i\ K(x_i, x_q) + b$$

#### Polynomial kernel 

$$K(x_i, x_j) = (x_i^Tx_j + c)^d = x_i'^Tx_j' $$

- $x_i'$ is of a higher dimension than $x_i$ and hence is a feature transformation from smaller dimension to larger dimension

#### Radial Basis Function Kernel : 

$$K_{RBF}(x_i,x_j)= exp(\frac{-||x_i-x_j||^2}{2\sigma^2})$$

Similarity between KNN and RBF Kernel 
- For same value of $\sigma$  as $distance \ \  d_{12} = ||x_i-x_j||^2$ increases $K_{RBF}$ (similarity) decreases
- For same value of $d_{12}$ as $\sigma$ increases $K_{RBF}$  (similarity) increases  
- Increase in $\sigma$  is same as increase in $K$ for K-NN
- RBF SVM is nice approximation of KNN with lower computational complexity

#### Domain Specific Kernels
- String Kernels : Text Classification 
- Genome Lernels
- Graph Based Kernels

## Training and Runtime 
### Optimization Algorithms
- Stochaistic Gradient Decent 
- Sequential Minimal Optimization - **libsvm** 

### Time Complexity
- Training Time complexity ~ $O(n^2)$ for Kernel SVMs ($n$ is no of traning points) Very High 
- Run Time complexity ~ $O(kd)$ for Kernel SVMs ($k$ is no of support vector points)

### Nu-SVM
Instead of hyperparameter $C$ another  hyperparameter $nu$ is used which has the property:

- $nu \ge fraction \ of\  errors$
- $nu \le fraction \ of\  support \ vectors$

So $nu$ can control the error  but does not control number of Support Vectors

### Outlier impact 
- Very little impact of outlier as only support vectors are used in calculation 
- For RBF with small $\sigma$  can be affected by outliers, similar to KNN with small K is affected by outliers

## Cases
- Good Case 
	- We know the right Kernel 
	- High Dimention Data 
- Bad Case 
	- $n$ is large imples high Training time 
	- $k$ (No of Support vector) is large 

#### Good Questions and Refernces

1. [A tutorial on support vector regression](https://alex.smola.org/papers/2004/SmoSch04.pdf)
2. [svm-skilltest](https://www.analyticsvidhya.com/blog/2017/10/svm-skilltest/)
3. [Give some situations where we use an SVM over a RandomForest and vice-versa](https://datascience.stackexchange.com/questions/6838/when-to-use-random-forest-over-svm-and-vice-versa)
4. [What is convex hull ?](https://en.wikipedia.org/wiki/Convex_hull)
5. What is a large margin classifier?
6. Why SVM is an example of a large margin classifier?
7. SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)
8. What is the role of C in SVM?
9. In SVM, what is the angle between the decision boundary and theta?
10. What is the mathematical intuition of a large margin classifier?
11. What is a kernel in SVM? Why do we use kernels in SVM?
12. What is a similarity function in SVM? Why it is named so?
13. How are the landmarks initially chosen in an SVM? How many and where?
14. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
15. What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)
16. How does the SVM parameter C affect the bias/variance trade off? (Remember C = 1/lambda; lambda increases means variance decreases)
17. How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?
18. Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)
19. Logistic regression vs. SVMs: When to use which one? ( Let's say n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, If n is small and m is intermediate, SVM with Gaussian kernel, If n is small and m is massive, Create or add more features then use log. Reg. or SVM without a kernel)