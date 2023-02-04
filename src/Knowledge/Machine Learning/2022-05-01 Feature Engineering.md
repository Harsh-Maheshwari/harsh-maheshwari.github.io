---
title: Feature Engineering
date: 2022-01-02 00:00:00
description: Feature Engineering
tags: 
 - Machine Learning
 - Data Science
 - Feature Engineering
---

!!! note 
	Before creating new feature research the domain and do some literature review

## Handaling different data

### Text Data 

#### Text Processing
- Remove Html characters, punctuation
- Stop word removal
- Lower case conversion
- Stemming : [ Porter Stemmer](https://www.nltk.org/_modules/nltk/stem/porter.html), [Snowball Stemmer](https://www.nltk.org/api/nltk.stem.snowball.html?highlight=snowball%20stemmer)
- Lemmitizatiom : break a sentence into word  [Ref1](https://stackoverflow.com/questions/1787110/what-is-the-true-difference-between-lemmatization-vs-stemming), [Ref2](https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/) 

#### Text to Vectors
#####  Bag of Words (BOW) 
- Semantic meaning of word is lost
- Binary/Boolean BOW
- Count BOW
- Uni-gram/ Bi-gram/N-gram

##### TF-IDF Term Frequency Inverse Documents Frequency  
- Semantic meaning of word is lost [Ref1](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) [Ref2](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089)

##### Word2Vec
- [Ways to train a Word2Vec Dictionary](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/):  CBOW & Skipgram
- CBOW predicts focus word given context words using an autoencoder model 
- Skipgram predicts context word  given focus word using an autoencoder model with multiple softmax (=number of context words) (Computationally more expensive to train, but can work with smaller data and infrequent words)
- Algorithmic optimisations for Skipgram and CBOW
	- Hieraarchial Softmax - No of softmax needed = $log_2(Vocab\  Size)$
	- Negatice Sampling - Update only a sample (based on frequency) of words including the target word
- In Word2Vec or Avg Word2Vec, Semantic information is learned by the vector 
- References : [Word Embedding](https://www.tensorflow.org/text/guide/word_embeddings)  | [Tensorflow Reference](https://www.tensorflow.org/tutorials/text/word2vec)  |  [TFIDF Weighted Word2Vec](https://medium.com/analytics-vidhya/featurization-of-text-data-bow-tf-idf-avgw2v-tfidf-weighted-w2v-7a6c62e8b097)

##### Positional Embedding
- [Semantic and  information about the Position in a sentence is captured](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [Keras Position Embedding](https://keras.io/api/keras_nlp/layers/position_embedding/)
- Transformers  : Absolute Or Relative [Positional Embedding](https://theaisummer.com/positional-embeddings/#positional-encodings-vs-positional-embeddings)

#####  BERT & DistillBERT
- Can be used as replacement for word2vec

!!! note 
	Transformers are more efficient in parallel processing than LSTMs [Reference 1](https://voidful.medium.com/why-transformer-faster-then-lstm-on-generation-c3f30977d747#:~:text=That's%20all%20for%20transformer%20model,neural%20network%20such%20as%20LSTM.), [Reference 2](https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen) 

### Categorical Data
- One hot encoding
- [Mean value replacement or response coding ](https://thierrymoudiki.github.io/blog/2020/04/24/python/r/misc/target-encoder-correlation)

### Time Series & Wave Form Data 
#### Moving Window
- Within a Moving Window of width $w$ calculate mean, std dev, median, quantiles, max, min, max - min,  local minimas,  local maximas, zero crossing

#### Fourier Transform
Any repeating pattern with Amplitude $A$, Time Period $T$, Frequency $F$ and phase $\phi$ can be broken down as sum of sine and cosines waves.  Fourier Transform is used to convert a wave from `Time domain` to `Frequency domain` which is very insightful in repeating patterns

![Oscillating_sine_wave](Oscillating_sine_wave.gif) ![FFT_Time_Frequency_View](/Assets/img/FFT_Time_Frequency_View.png)


##### Discrete Fourier Transform (DFT)
1. The **input** is a sequence of numbers of the original variable (_one value per time step_)
2. The **output** is _one value of amplitude (or signal strength) for each frequency_. These are represented by the **Fourier Coefficients**.  
3. This new series is computed using the Fourier formula:

   
$$X_k = \sum_{n=0}^{N-1} x_ne^{-2 \pi ikn/N}$$
 
> Now we obtain the frequencies that are present in the variable.  Each of the values in the outcome series is the strength of a specific frequency.  If the amplitude of a frequency is high, then that seasonality is important in the orginal time series (or waves). There exists an Computational optimized form of DFT called Fast Fourier Transform. It is computed using the **Cooley-Tukey FFT algorithm.**
   

### Graph Data
#### Node Level Features
##### Node InDegree and Node OutDegree 
##### Eigenvector Centrality
##### Clustering Coefficient
##### DeepWalk
##### Graph coloring
##### HOPE
##### [Page Rank](https://en.wikipedia.org/wiki/PageRank)
Adar Index
Kartz Centrality
Shortest Path
Connected-component
HITS Score


#### Graph Level Features
##### Adjacency Matrix
##### Laplacian Matrix
##### Bag of Nodes
##### Weisfeiler-Lehman Kernel
##### Graphlet Kernels
##### Path-based Kernels
##### [GraphHopper kernel](https://ysig.github.io/GraKeL/latest/kernels/graph_hopper.html), [Neural Message Passing](https://arxiv.org/abs/1704.01212),  [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

#### Neighbourhood Overlap Features
##### Local Overlap Measures
##### Global Overlap Measures
#####
#####
#####
#####

### Sequence Data 

### Image Data
- [Colour Image Histogram](https://en.wikipedia.org/wiki/Image_histogram) - [Tutorial](https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/)
- [Edge Histogams](https://kenndanielso.github.io/mlrefined/blog_posts/14_Convolutional_networks/14_2_Edge_histogram_based_features.html)
- [Haar Features](https://en.wikipedia.org/wiki/Haar-like_feature)
- [SIFT Features](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) : Very useful in Image Search with properties like scale invariance and rotation invariance

### Graph Data

## Feature Engineering
- Feature Binning and Indicator variables 
	- Depending on domain knowledge sometime it might be useful to convert real valued / categorical feature to bucketed/bined feature based on some rule
	- To find the Binning thresholds / rule for a real valued variable $X$ train a decision tree using $X$ and target $Y$ and get the threshold from the decision trees (trained on entropy)
- Interaction Variables : 
	- Logical $N$ way Interaction between $N$ variables created using Decision Trees of Depth $N$
	- Numerical $N$ way Interaction between $N$ variables created using Mathamteical Operations between these features
- Feature Orthogonality : More orthogonal/different/uncorrelated features are better for learning a model
- Slicing Features : Features that help us divide the data into separate data generating process Eg: Device Feature : {Desktop, Mobile, Tablet} The way a customer would a product on mobile would be different and on computer would be different



## References

1. [Regular Expression Blog](https://pymotw.com/2/re/) 
2. [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html#documentation)
3. [Sliding Window Discrete Fourier Transform](/Assets/pdf/Sliding_Window_Discrete_Fourier_Transform.pdf)
4. https://kenndanielso.github.io/mlrefined/


