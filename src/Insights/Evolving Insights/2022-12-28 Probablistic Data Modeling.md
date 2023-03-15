---
title: Probablistic Data Modeling
date: 2022-12-28 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---

## Count Data Models

- Consumer demand : for example the number of products that a consumer buys on Amazon 
- Recreational data : the number of trips taken per year 
- Family economics : The number of children that a couple has
- Health demand : the number of doctor visits in a reasonably big geographical area

Dependent variable is counts which is a non-negative integer so Y would be equal to 0 1 2 3 4 and so on.  These are indeed numbers but the sample is concentrated on a few very small discrete values like for example you can say well how many grains of rice do you consume that's also could theoretically be counted but it's just so much that you may not want to use how many grains of rice but instead how many pounds of rice or some like that and move more toward continuous variable models rather than the discrete variable models so make sure that your when you have data that is count data the numbers are relatively small you know most of the the sample say within 10 20 30 as far as as numbers and you have very few observations beyond that if you have numbers in the thousands and so on then perhaps this is not a very good model for the use case. 

The number of children that a couple has that is definitely a discrete variable however there may be situations in which that could be turned into a continuous variable.  For example "how many children per couple" are in each of the countries and in this way you know you have like a country with 2.3 children per couple in a given country.This again becomes a continuos distribution not good for count data models

### Count data dependent variable and its properties

- The dependent variable is counts (a non-negative integer): y = 0, 1, 2, 3, 4, ...
- An event can occur any number of times during a time period but the sample is concentrated on a few small discrete values.
- Events occur independently. In other words, if an event occurs, it does not affect the probability of another event occurring in the same time period.
- We study the factors affecting the average number of the dependent variable

### Poisson model
The Poisson model predicts the number of occurrences of an event. The Poisson model states that the probability that the dependent variable Y will be equal to a certain number y is :


$$p(Y=y)= {e^{-\mu}\mu^{y} \over y!}$$

where $\mu$ is the intensity or rate parameter

#### Properties of the Poisson distribution

- **Equidispersion property** of the Poisson distribution i.e the equality of mean and variance. 
	- $E(y|x)=var(y|x)=\mu$
	- Equidispersion property is a restrictive property and often fails to hold in real world examples, i.e., there is “overdispersion” in the data. In this case, use the negative binomial model.   
- **Merging Independent Poisson Processes** : 
	* Let $N_1(t), N_2(t), ... , N_m(t)$  be $m$ independent Poisson processes  
	* Let the rates be $λ_1, λ_2, ...,  λ_m$  
	* Let  $N(t) = N_1(t) + ... + N_m(t)$, for all $t∈[0,∞)$ 
	  
	  Then, $N(t)$ is a Poisson process with rate $λ_1+λ_2+ ... +λ_m$

- **Skellam Poisson Relation** : 
	- Let $N_1(t), N_2(t)$  be 2 independent Poisson processes with rates $λ_1, λ_2$
	- Let $S(t) = N_1(t) - N_2(t)$ ,for all $t∈[0,∞)$ 
	  
	  Then $S(t)$ is a Skellam distribution with $rate1 =  λ_1$ and $rate2 =  λ_2$

- **Marginal effects for the Poisson model** : One unit increase in x will increase/decrease the average number of the dependent variable events by the marginal effect. It is given by:
  
$$\frac{\partial E(y|x)}{\partial x_j} =\beta_jexp(\mathbf{x}_i^{'}\beta)$$


- Excess zeros problem of the Poisson distribution: there are usually more zeros in the data than a Poisson model predicts. In this case, use the zero-inflated Poisson model.
- The probability of an event occurring is proportional to the length of the time period. For example, it should be twice as likely for an event to occur in a 2 hour time period than it is for an event to occur in a 1 hour period.


### Negative binomial model
The negative binomial model is used with count data instead of the Poisson model if there is
overdispersion in the data. Unlike the Poisson model, the negative binomial model has a less restrictive property that the variance is not equal to the mean $\mu$
$$var(y|x) = \mu + \alpha\mu^2$$
Where  $\alpha$  is the overdispersion parameter and $\mu$ is the intensity or rate parameter. Another functional form is $var(y|x) = \mu + \alpha\mu$,but this form is less used

#### Test for overdispersion
We estimate the negative binomial model which includes the overdispersion parameter $\alpha$ and test if $\alpha$ is significantly different than zero.

- We have three cases for $H_0: \alpha = 0$  or  $H_a: \alpha \ne 0$

	- When $\alpha = 0$, the Poisson model. 
	- When $\alpha > 0$ , overdispersion (frequently holds with real data).
	- When $\alpha < 0$, underdispersion (not very common).

#### Incidence rate ratios (irr)
- The incidence rate ratios report $exp(b)$ rather than $b$.
- Interpretation of the incidence rate ratios: irr=2 means that for each unit increase in x, the
expected number of y will double.

### Hurdle or two-part models 

The two-part model relaxes the assumption that the zeros (whether or not there are events) and
positives (how many events) come from the same data generating processes. Example: different factors may affect whether or not you practice a particular sport and how many times you practice your sport in a month. We can estimate two-part models similar to the truncated regression models. If the process generating the zeros is $f_1$ and the process generating the positive responses is $f_2$ then the two-part hurdle model is defined by the following probabilities:

$$ g(y) = f_1(0) \ \ if \ \ y=0$$

$$g(y) = \frac{1-f_1(0)}{1-f_2(0)}f_2(y) \ \ if\ \ y\ge1$$

The model for the zero versus positive responses is a binary model with the specified
distribution, but we can estimate it with the probit/logit model.

### Zero inflated models 

The zero-inflated model is used with count data when there is an excess zeros problem. Example: you either like hiking or you do not. If you like hiking, the number of hiking trips
you can take is 0, 1, 2, 3, etc. So you may like hiking, but may not take a trip this year. We
are able to generate more zeros in the data.  The zero-inflated model lets the zeros occur in two different ways:  

1. As a realization of the binary process (z=0) 
2. As a realization of the count process when the binary variable z=1.



If the process generating the zeros is $f_1$ and the process generating the positive responses is
$f_2$ then the zero-inflated model is:

$$g(y)=f_1(0)+(1-f_1(0))f_2(0) \ \ if \ \ y = 0$$

$$g(y)=(1-f_1(0))f_2(y) \ \ if \ \ y \ge 1$$

