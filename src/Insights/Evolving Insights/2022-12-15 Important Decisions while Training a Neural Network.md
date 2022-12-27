---
title: Important Decisions while Training a Neural Network
date: 2022-12-15 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---

## General Advice
General advice while training neural networks : 

- Increase complexity one at a time and always provide a hypothesis on how should the model react to the said change in complexity. And verify if the hypothesis is True or False and why.
- Understand as many paterns as possible from the data using EDA and figure out which pattern is the model able to learn and why
- Before extensive experimentation get a complete pipeline executed for small data and small model.  The pipeline should be from getting raw data to finalising the techincal and business reports
- Set up a full training + evaluation + testing skeleton and gain trust in the pipeline execution via a series of experiments with explicit hypotheses, losses and metrics visualizations, model predictions. 
- Follow Andrej Karpathy's [guide](http://karpathy.github.io/2019/04/25/recipe/) on basic steps

## Data Decisions
### Feature Engineering
- Select features and remove any that may contain patterns that won't generalize beyond the training set and cause overfitting
- Scaling your features: For faster convergence all  features should have a similar scale before using them as inputs to the neural network.  

## Network Architechture Decisions
### Input neurons
- For tabular dataset input vector is a combination of one input neuron per feature, So the shape of the input vector is the number of features selected for training
- For images dataset input vector is the dimensions of your image
- For text dataset input vector is decided by the text to vector embedding

### Output neurons
Number of output neurons depends on the type and number of predictions
- For single regression target this is a one value
- For multi-variate regression, it is one neuron per predicted value, For bounding boxes we have 4 regression values one for each bounding box property : height, width, x-coordinate, y-coordinate
- For binary classification we use one output neuron which represents the probability of the positive class. 
- For multi class classification, we have one output neuron per class, and use the softmax activation function on the output layer to ensure the final probabilities sum to 1. 

### Hidden Layers
- No of hidden layers and no of nuerons per hidden layer: Decided by hit and trial, Start with either a too big or a too small network and adjust incrementally untill the model is neither overfitting nor underfitting.
- General Recommendation : Start with 1-5 layers and 1-100 neurons (same number of neurons for all hidden layers) and slowly adding more layers and neurons until you start overfitting. Usually we get more performance boost from adding more layers than adding more neurons in each layer.
- When chossing a smaller number of layers/neurons, the network will not be able to learn the underlying patterns in your data and thus be useless. An approach to counteract this is to start with a huge number of hidden layers and neurons and then use dropout and early stopping to let the neural network size itself down for you
- For many problems in image or speech domain there are pre-trained models (YOLO, ResNet, VGG) that allow you to use large parts of their networks, and train your model on top of these networks to learn only the higher order features. 
- Manytimes having a large first layer and following it up with smaller layers will lead to better performance as the first layer can learn a lot of lower-level features that can feed into a few higher order features in the subsequent layers.

### Skip Connections

### No of Parameters 
- Calculate the number of parameters in the model and compare with the number of data points you have. (Just keep the comparision in mind).   

### Activation Functions 
- [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf)
- The performance from using different hidden layer activations improves in this order (from lowest→highest performing): logistic → tanh → ReLU → Leaky ReLU → ELU → SELU. 
- ReLU is the most popular activation function , But  [ELU](https://arxiv.org/pdf/1511.07289.pdf) or [GELU](https://arxiv.org/pdf/1606.08415.pdf) are on the rise
- To combat specific problems:
	- RReLU : To combat neural network overfitting
	- PReLU: For massive training sets
	- leaky ReLU: Reduce latency at runtime and fast inference times
	- ELU: If your network doesn't self-normalize
	- SELU: For an overall robust activation function
- Regression output actiation 
	- softplus : For positive prediction 
	- _scale_\*tanh : For predictions in range  _-scale_ and _scale_
	- _scale_\*sigmoid : For predictions in range  0 and _scale_

### Weight Initialization 
- When using softmax, logistic, or tanh, use [Glorot initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) Mostly default used in Tensorflow 
- When using ReLU or leaky RELU, use [He initialization](https://arxiv.org/pdf/1502.01852.pdf)
- When using SELU or ELU, use [LeCun initialization](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### Vanishing and Exploding Gradients
- **Vanishing Gradients** : when the backprop algorithm propagates the error gradient from the output layer to the first layers, the gradients get smaller and smaller until they're almost negligible when they reach the first layers. This means the weights of the first layers aren't updated significantly at each step.
- **Exploding Gradients** : when the gradients for certain layers get progressively larger, leading to massive weight updates for some layers as opposed to the others
- **Gradient Clipping** is a great way to reduce gradients from exploding, specially when training RNNs. Simply clip them when they exceed a certain value. Use clipnorm instead of clipvalue, which allows you to keep the direction of your gradient vector consistent. Clipnorm contains any gradients who's l2 norm is greater than a certain threshold.
- **BatchNorm** simply learns the optimal means and scales of each layer's inputs. It does so by zero-centering and normalizing its input vectors, then scaling and shifting them. It also acts like a regularizer which means we don't need dropout or L2 reg.
- Using **BatchNorm** lets us use larger learning rates (which result in faster convergence) and lead to huge improvements in most neural networks by reducing the vanishing gradients problem. The only downside is that it slightly increases training times because of the extra computations required at each layer.
- **Early Stopping** lets you live it up by training a model with more hidden layers, hidden neurons and for more epochs than you need, and just stopping training when performance stops improving consecutively for n epochs. It also saves the best performing model for you.
- **Dropout** gives you a massive performance boost (~2% for state-of-the-art models). All dropout does is randomly turn off a percentage of neurons at each layer, at each training step. This makes the network more robust because it can't rely on any particular set of input neurons for making predictions. The knowledge is distributed amongst the whole network. 
- In **Dropout** around 2^n (where n is the number of neurons in the architecture) slightly-unique neural networks are generated during the  training process, and ensembled together to make predictions.
- A good **Dropout Rate** is between 0.1 to 0.5; 0.3 for RNNs, and 0.5 for CNNs. Use larger rates for bigger layers. Increasing the dropout rate decreases overfitting, and decreasing the rate is helpful to combat under-fitting.
- Read [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/abs/1801.05134) before using Dropout in conjunction with BatchNorm.

## Optimisation and Training Decisions

### Optimizers
- Adam/Nadam are usually good starting points, and tend to be quite forgiving to a bad learning late and other non-optimal hyperparameters.
- Use Stochastic Gradient Descent if you care deeply about quality of convergence and if time is not of the essence.
- If you care about time-to-convergence and a point close to optimal convergence will suffice, experiment with Adam, Nadam, RMSProp, and Adamax optimizers


### Batch Size
- Large batch sizes can be great because they can harness the power of GPUs to process more training instances per time. [OpenAI has found](https://openai.com/blog/science-of-ai/) larger batch size (of tens of thousands for image-classification and language modeling, and of millions in the case of RL agents) serve well for scaling and parallelizability.
- According to [this paper](https://arxiv.org/abs/1804.07612) by Masters and Luschi, the advantage gained from increased parallelism from running large batches is offset by the increased performance generalization and smaller memory footprint achieved by smaller batches. They show that increased batch sizes reduce the acceptable range of learning rates that provide stable convergence. Their takeaway is that smaller is, in-fact, better; and that the best performance is obtained by mini-batch sizes between 2 and 32.
- If you're not operating at massive scales, start with lower batch sizes and slowly increasing the size and monitoring performance to determine the best fit.
- [Batch vs Mini-batch vs Stochastic Gradient Descent](https://medium.datadriveninvestor.com/batch-vs-mini-batch-vs-stochastic-gradient-descent-with-code-examples-cd8232174e14) 

### Learning Rate And Momentum
- Use a constant medium learning rate until you've trained all other hyper-parameters and implement learning rate decay scheduling at the end.
- Start with a very low values (10^-6) and increase by a factor of 10 until it reaches a very high value (e.g. 10). Measure your model performance (vs the log of your learning rate) to determine which rate served you well for your problem. 
- The best learning rate is usually half of the learning rate that causes the model to diverge.
- [Learning Rate finder](https://arxiv.org/abs/1506.01186) method proposed by Leslie Smith. It an excellent way to find a good learning rate for most gradient optimizers (most variants of SGD) and works with most network architectures.
- With learning rate scheduling we can start with higher rates to move faster through gradient slopes, and slow it down when we reach a gradient valley in the hyper-parameter space which requires taking smaller steps.
- Gradient Descent takes tiny, consistent steps towards the local minima and when the gradients are tiny it can take a lot of time to converge. Momentum on the other hand takes into account the previous gradients, and accelerates convergence by pushing over valleys faster and avoiding local minima.
- In general you want your momentum value to be very close to one. 0.9 is a good place to start for smaller datasets, and you want to move progressively closer to one (0.999) the larger your dataset gets.
- Setting nesterov=True lets momentum take into account the gradient of the cost function a few steps ahead of the current point, which makes it slightly more accurate and faster.

![](/Assets/img/learning rate vs epochs.png)


### Number of epochs
- Start with a large number of epochs and use early stopping to halt training when performance stops improving.

### Loss Function
#### Mean Squared Error  vs Huber loss 
- Use MSE if the data does not have significant number of outliers
- Use Huber loss if the data does have significant number of outliers
#### Mean Absolute Error 
- More robust to outliers in data than the Mean Squared Error.
- Does not scale with magnitude output.  
#### Mean Absolute Percentage Error 
- It cannot be used if there are zero or close-to-zero values because there would be a division by zero or values of MAPE tending to infinity.
- MAPE puts a heavier penalty on negative errors than on positive errors as stated in Accuracy measures: theoretical and practical concerns - [REF](https://www.sciencedirect.com/science/article/abs/pii/0169207093900793) As a consequence, when MAPE is used to compare the accuracy of prediction methods it is biased in that it will systematically select a method whose forecasts are too low. This issue can be overcome by using an accuracy measure based on the logarithm of the accuracy ratio (the ratio of the predicted to actual value). This leads to superior statistical properties and also leads to predictions which can be interpreted in terms of the geometric mean.
- To overcome these issues with MAPE, there are some other measures proposed in literature:
	- [Mean Absolute Scaled Error](https://en.wikipedia.org/wiki/Mean_Absolute_Scaled_Error "Mean Absolute Scaled Error") (MASE) , 
	- [Symmetric Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Symmetric_Mean_Absolute_Percentage_Error "Symmetric Mean Absolute Percentage Error") (sMAPE)
	- [Mean Directional Accuracy (MDA)](https://en.wikipedia.org/wiki/Mean_Directional_Accuracy_(MDA) "Mean Directional Accuracy (MDA)")
	- Mean Arctangent Absolute Percentage Error (MAAPE): MAAPE can be considered a _slope as an angle_, while MAPE is a _slope as a ratio_

### Classification
- Cross entropy or Sparse Categorical Cross Entropy

#### References

- [kaggle](https://www.kaggle.com/code/lavanyashukla01/training-a-neural-network-start-here/notebook)
- [EfficientNets](https://arxiv.org/pdf/1905.11946.pdf) to scale your network in an optimal way.
- Read [A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS](https://arxiv.org/pdf/1803.09820.pdf) for an overview of some additional learning rate, batch sizes, momentum and weight decay techniques. - [Blog](https://jithinjk.github.io/blog/disciplined_nn_approach.md.html) 
- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) shows that better generalization can be achieved by averaging multiple points along the SGD's trajectory, with a cyclical or constant learning rate.
- [Images](https://cs231n.github.io/neural-networks-3/)