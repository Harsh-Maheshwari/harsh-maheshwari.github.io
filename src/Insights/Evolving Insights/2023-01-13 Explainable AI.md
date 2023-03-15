---
title: Explainable AI
date: 2023-01-13 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---

## Research Papers, Blogs & Resources
1. [Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)
2. [Integrated Gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
3. [Robustness of Interpretability Methods](https://arxiv.org/pdf/1806.08049.pdf)
4. [Interpretable Machine Learning Web Book](https://christophm.github.io/interpretable-ml-book/index.html)
5. [LIME TDS 1](https://towardsdatascience.com/lime-how-to-interpret-machine-learning-models-with-python-94b0e7e4432e) | [LIME Blog](https://towardsdatascience.com/interpreting-image-classification-model-with-lime-1e7064a2f2e5) | [LIME Text Explain](https://towardsdatascience.com/what-makes-your-question-insincere-in-quora-26ee7658b010)

### Python Libraries
1. [explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/index.html)
2. [omnixai](https://opensource.salesforce.com/OmniXAI/latest/index.html)
3. [InterpretML](https://interpret.ml/docs/getting-started)
4. [ELI5](https://eli5.readthedocs.io/en/latest/)
5. [Shapash](https://shapash.readthedocs.io/en/latest/)
6. [LIME](https://github.com/marcotcr/lime)
7. [SHAP](https://shap.readthedocs.io/en/latest/)

## Local vs Global Interpretability 

- Global feature importance comes from understanding how the model breaks down the global space/dimension and tries to explain how are all the points in data predicted
- Local feature importance comes from understanding how the model behaves around the locality of current query point
- Within the neighbourhood of the query point create a surrogate model which aproximates the original model in that neighbourhood. This surrogate model can be a basic linear model or a Decision Trees, which have easily interpretable feature importance
- **Interpretable representation** : Conversion of a feature of dimension d to a binary feature of dimension d'. Ex : for text use Bag of Word, for image use super pixel, for real values use real valued binning
- The Surrogate model will use these Interpretable representation as input rather than the original input

