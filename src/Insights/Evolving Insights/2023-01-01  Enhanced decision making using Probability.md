---
title: Enhanced decision making using Probability
date: 2023-01-01 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---

## Concepts in Probability

Probability computes the likelihood of an event from a random variable. Let's consider $x$ is a random variable then the probability of an event $X$ can be expressed as $P(x=X)$. For example, flipping a coin can be represented as random variable $x$ then the probability of an event getting head, $P(x=head)$ will be $0.5$

### Joint and Conditional Probabilities 
Consider two random variables $x$ and $y$  and  pick two events $X$ and $Y$ from these random variables to define the following probabilities:

1.  **Conditional probability** computes the probability of occurring of event $X$ given that event $Y$ has occurred and is denoted by $P(X|Y)$.
2.  **Joint probability** is defined as the probability of both events $(x=X \ and \  y=Y)$ jointly and is calculated as $P(X \cap Y) = P(X|Y)*P(Y)$

!!! note "Given $x$ as a set of training data and $y$ as corresponding label"
	- **[Discriminative Models](https://en.wikipedia.org/wiki/Discriminative_model)** learn the conditional distribution $P(y|x): x → y$  and during inference, $P(y|x)$ will compute the likelihood of that sample belonging from the given class.
	- **[Generative models](https://en.wikipedia.org/wiki/Generative_model)** learn the joint distribution $P(x,y)$, which can be decomposed in conditionals using the definition of the joint probability $P(x, y) = P(x|y) P(y)$. Then these models use the marginal probability of classes and then learn $P(x|y)$.

### Bayes Theorem
Bayes theorem allows manipulation between conditional probabilities. This theorem can be derived using the commutative property of joint probability, which means:

$$P(X \cap Y) = P(Y \cap X)$$

$$P(X|Y)P(Y) = P(Y|X)P(X)$$

$$P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}$$

In generative models, evaluating $P(x|y)P(y)$ will be easier compared to $P(y|x)P(x)$ given that $y$ is sampled from a tractable density.

### Entropy

Entropy is similar to the concept of _information_, which is defined as the number of bits required to transmit an event. Low probability events have lesser predictability therefore they have higher information and vice versa. Entropy measures the uncertainty of a system and is defined as the number of bits required the transmit a random event from a probability distribution. Similar to information, entropy can be defined as:

$$H(x) = - \sum_{X \ \epsilon \ x} P(X)*log(P(X))$$

### Cross entropy

Cross entropy is based on the concept of Entropy and is defined between two probability distributions. Let us consider two probability distributions $p$ and $q$. Then the cross-entropy of $q$ from $p$ is the number of additional bits to represent an event using $q$ instead of $p$ and can be formulated as:

$$CE(p,q) = - \sum_{X \ \epsilon \ x} p(X)*log(q(X))$$


In DL, cross-entropy is the default loss function for both discriminative and generative models. Let's take an example of binary classification (cat and dog classification), where $p$ is the given distribution for the labels with probabilities 0 or 1. We would like to approximate the distribution $q$ such that an event from $q$ can represent $p$ with minimum number of bits. Since $X$ can have two values, therefore:

$$CE(p,q) = - p(X=0)*log(q(X=0)) - p(X=1)*log(q(X=1))$$

if $p$ represents the labelled data $y_i$ and $q$ represents the predicted output $prob_i$, then above equation can be modified to

$$CE(p,q) = \sum_{\forall_i} -y_i*log(prob_i) -(1-y_i)*log(1-prob_i)$$

In the above equation, if $q(X=0) = prob_i$ then $q(X=1) = 1- prob_i$ because of the binary classification.

## Measure distances between probabilities distributions


### Kullback - Leibler divergence

Kullback - Leibler Divergence also called relative entropy calculates a distance metric between two PDFs or divergence between two probabilities distributions. KL-divergence between $p$ and $q$ (two probability distributions) computes the number of extra bits to represent an event using $q$ instead of $p$. While cross-entropy computes an average number of total bits to represent an event from $q$ instead of $p$. KL divergence is diffferentiable and can be written as:

$$ D_{KL}(p||q) = - \int_x p(x) log(\frac{q(x)}{p(x)})$$

$$ D_{KL}(P||Q) = - \sum_x p(x) log(\frac{q(x)}{p(x)})$$

The above equation can be expanded and rewritten in the form of cross-entropy and entropy:

$$CE(p,q) = D_{KL}(p||q) + H(p)$$

1. When $p$ (target distribution) and $q$ (predicted distribution) are the same then KL divergence will be zero, i.e. the lower value of KL divergence indicates the higher similarity between two distributions.
2. KL divergence and cross-entropy both are not symmetrical (a con which is ignored many times)
3. Cross-entropy is mostly used as the loss function and from an optimization point of view, KL divergence and cross-entropy will be the same because the entropy term $H(p)$ is a constant and will be zero during the derivative calculations.

### Jensen Shannon (JS ) divergence

Jensen Shannon divergence also quantifies the difference between two probability distributions and extends KL-divergence to calculate the symmetrical measure. JS-divergence between $p$ and $q$ is:

$$D_{JS}(p||q) =  \frac{1}{2}D_{KL}(p||\frac{p+q}{2}) + \frac{1}{2}D_{KL}(q||\frac{p+q}{2})$$

1.  JS-divergence computes the symmetrical measure.
2.  If we use base-2 logarithm then the above measure is a normalized version of KL divergence, with scores between 0 (same) and 1 (completely different).
3. JS-divergence is an improved version of KL-divergence with symmetric measures and normalized outcomes
4. Normalization provides better stability during the loss function optimization and therefore JS-divergence is prefered for the complicated tasks (GAN) compared to KL-divergence


### Wasserstein metric

The Wasserstein metric calculates the distance between two probability distributions and is termed as Earth Mover’s distance, which measures the minimum energy cost of moving a pile of dirt in the shape of one probability distribution to the shape of the other distribution. The Wasserstein metric between $p$ and $q$ is defined as

$$W(p,q) = \underset{\gamma \ \sim \ \prod (p,q)}{\inf} \  \mathbb{E}_{(x,y) \ \sim \ \gamma} * [||x-y||]$$

In the above equation, the product term $\prod (p,q)$ represents the set of joint probabilities. The term $\gamma(x, y)$ is a joint probability and measures the amount of dirt that should be moved from $x$ to $y$ for $x$ to follow the same distribution as $y$. Therefore, the total amount of moved dirt would be $\gamma(x, y)$ with the moving distance of $||x-y||$ and further the cost will be $\gamma(x, y).||x-y||$ for a given set of $x$ and $y$. The term $\inf$ (infimum) indicates that the above equation will always provide the smallest cost. Note Marginals of $\prod (p,q)$ will be $p$ and $q$

1. Wasserstein metric is a metric, not a divergence which means it follows all three metric properties (positive, symmetric and triangle inequality) that the Wasserstein metric is more stable and smooth in the optimization problems.
2. It is able to compute the distance between two distributions even if they are not in the same probability space, unlike KL-divergence

### KL divergence vs JS divergence vs Wasserstein metric
Consider $p$ , $q$ two uniform distributions centred at $0$ and $\theta$. Basically both distributions $p$ and $q$ are not overlapping and have a distance of $\theta$ between them, which is a very simple example of a non-sharing probability space.  The values of all the metrics discussed is given below

$$ D_{KL}(p||q) = \infty$$

$$ D_{KL}(q||p) = \infty$$

$$ D_{JS}(p||q) = log(2)$$

$$ W(p,q) = |\theta|$$

- KL-divergence between these two distributions will be $\infty$ because of non-overlapping and the denominator part will be zero
- JS-divergence will be constant $log(2)$ irrespective of the value of $\theta$ as one of them will be zero at the given $x$
- Wasserstein metric depends on the distance between these two distributions which is desirable and considering the horizontal shifts irrespective of overlaps. Therefore, the Wasserstein metric is more practical and produces better results compared to JS-divergence 


## Kolmogorov–Smirnov test

**Kolmogorov–Smirnov test** _KS test_ is a [nonparametric test](https://en.wikipedia.org/wiki/Nonparametric_statistics "Nonparametric statistics") of the equality of continuous (or discontinuous), one-dimensional [probability distributions](https://en.wikipedia.org/wiki/Probability_distribution "Probability distribution") that can be used to compare a [sample](https://en.wikipedia.org/wiki/Random_sample "Random sample") with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test). KS Statistic is not differentiable

$$KS \ Stat = Max (|p'(x)-q'(x)|)$$

where $p'$ = CDF of $P(x)$

## How to measure if two distributions are similar or not ?
- There are many tests and metrics like KS-test, JS-divergence, KL-divergence, Wasserstein metric to get an idea about similarity of two distributions. But all of these have some or the other limiltation. 
- The most robust way to estimate similarity is to train a model with each component distribution given a class label, i.e. for the $i_{th}$ distrubution $D_i$  we can assign a distinct label $i$ in the target $y$. Then train a fairly complex model, If the log loss of the traied model is low then the model is able to distinguish between the distributions. and hence the distributions are different. If the log loss is on the higher side than the model is not able to distinguish between the distributions and hence the distributions are similar.

#### References
1. [basic probability](https://medium.com/@sunil7545/necessary-probability-concepts-for-deep-learning-557f75dd3bce)
2. [comparing distributions](https://medium.com/@sunil7545/kl-divergence-js-divergence-and-wasserstein-metric-in-deep-learning-995560752a53)
3. Arjovsky, Martin, Soumith Chintala, and Léon Bottou. “Wasserstein generative adversarial networks.” _International conference on machine learning_
4. Weng, Lilian. “From gan to WGAN.” _arXiv preprint arXiv:1904.08994_ (2019).