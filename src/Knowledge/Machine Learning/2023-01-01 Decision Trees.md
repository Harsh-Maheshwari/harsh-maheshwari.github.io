---
title: Decision Trees
date: 2023-01-01 00:00:00
description: Decision Trees
tags: 
 - Machine Learning
 - Decision Trees
---

## Interpretation of Decision Trees

- **Programatic Interpretation** : Decision Trees are nested if else based models. 
- **Geometric Interpretation** : Decision Surfaces for decision trees are Set of axis parallel hyperplanes which divide the space into cuboids or hyper cuboids 

## Information Theory and Entropy

Let $Y$  be a random variable taking valyes $y_1, y_2,y_3,y_4, ...$ , Then $H(Y)$,  Entropy of $Y$ is given as 

$$H(Y) = - \sum_i^k p(Y=y_i) \ log_b(p(Y=y_i))$$

- Entropy is maximum when all posible values of Random Variable $Y$ are equi probable
- More peaked distributions have less entropy  

### Gini Impurity vs Information Gain vs MSE Reduction

- Information Gain $IG$ = Entropy of parent – Weighted Average (weights based on number of data points in each child) Entropy of child nodes

$$IG (Y, var) = H_D(Y)  - \sum_i^K \frac{|D_i|}{|D|} H_{D_i}(Y)$$

- Gini Impurity  $I_G$ is used because in practice rather than Information Gain, because it does not contain a Log term which is computationally expensive 

$$ I_G(Y) = 1 - \sum_i^k(p(Y=y_i))^2$$

- For Regression problems MSE Reduction is used, where predicted value for a node is the mean of  values in the y true values in the node

$$MSE \ Reduction = MSE\ Parent - \sum_{childs} MSE\  Child$$

## Important Points
### Outliers and Depth
- As depth of the tree increases the tree overfits, As depth of the tree decreases the tree underfits   
- Outliers with high depth will make the tree unstabke and effect of overfitting is high 

### Feature standardisation
Feature standardisation is not required in Decision Trees because we are not using distances anywhere

### High Dimentional Categorical Feature
A problem with high dimentional categorical feature is that there would be very less examples in the child nodes and the tree would be very wide (if one hot encoded).  Lets say there are k target classes, then the solution is to replace each category $j$ in the high dimentional categorical feature with a vector $v$ of size k, where $v = \{P(y=i|category=j)\}_k$ . This is called [Mean value replacement or response coding ](https://thierrymoudiki.github.io/blog/2020/04/24/python/r/misc/target-encoder-correlation) To Smooth the probability values we apply [laplace smoothing](https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece). Note always use only the training data to get the response coded vectors. If the train data does not contains a category which is seen in cross validation data or test data Lalace smoothing will make it a constant value ( 1/k ) vector 

### Time Complexity
- Trainining Complexity : $O(nlog(n)d)$ , n = no of training examples , d = dimension 
- Runtime Complexity : $O(depth)$ - Low Latency even for big data sets

### Special Cases
- **Imbalanced Data** : Balance the data 
- **Large Dimensionality** : High train time complexity, so  do not use one hot encoding rather convert to probabilities for each class
- Given similarity matrx only, decision tress cannot work
- Logical feature interactions are already inbuilt in Decision Trees
- Feature Importance is calculated by calculating the average reduction in the entropy due to a feature 


## Reference and Questions
1. [Decision Trees Lecture Notes](https://homepage.cs.uri.edu/faculty/hamel/courses/2014/spring2014/csc581/lecture-notes/31-decision-trees.pdf)
2. How to Building a decision Tree?
3. Importance of Splitting numerical features.?
4. How to handle Overfitting and Underfitting in DT?
5. How to implement Regression using Decision Trees?
6. https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/
7. https://vitalflux.com/category/career/interview-questions/