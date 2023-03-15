---
title: Big Query Vs Tensorflow Transform
date: 2022-12-15 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---
## Data Transformation 

In this section we will see what are the google product which we can use for Data Transformation

### BigQuery
!!! Info annotate " What is BigQuery ?" 
	According to google : "BigQuery is Google Cloud's fully managed, petabyte-scale, and cost-effective analytics data warehouse that lets you run analytics over vast amounts of data in near real time. With BigQuery, there's no infrastructure to set up or manage, letting you focus on finding meaningful insights using standard SQL and taking advantage of flexible pricing models across on-demand and flat-rate options"

#### Notes about BigQuery
- Serverless Infrastructure with a Scale Friendly Pricing Structure. This also reduces chances of failure
- Its Flexible Architecture Speeds Up Queries
- SQL is very easy to work with and is quite popular
- Access the Data You Need on Demand

### TensorFlow Transform
!!! Info annotate " What is TensorFlow Transform ?"
	 TensorFlow Transform is a library for preprocessing input data for TensorFlow and it lets you define both instance-level and full-pass data transformations through data preprocessing pipelines. These pipelines are efficiently executed with [Apache Beam](https://beam.apache.org/) and they create as byproducts a TensorFlow graph to apply the same transformations during prediction as when the model is served.

#### Notes about TensorFlow Transform
- TensorFlow transform is a hybrid of **Apache Beam and TensorFlow**
- Infrastructure must be defined in Dataflow and costs would depend on the cluster size 
- Speed of execution depends highly on your Dataflow configurations 
- It can be used to perform almost all types of transformation and It is as easy as writing python code
- It is not yet possible to do Window aggregations transformation with tf.Transform

| Data preprocessing option              | Instance-level (stateless transformations)                                                                                                                                                                                                                                                        | Full-pass during training and instance-level during serving    (stateful transformations)                                                                                                                                                                                                                        | Real-time (window) aggregations during training and serving (streaming    transformations)                                                                                                                                                                          |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BigQuery (SQL)                         | Batch scoring: OK —the same transformation implementation is      applied on data during training and batch scoring. Online prediction: Not recommended —you can process training data,      but it results in training-serving skew because you process serving data      using different tools. | Batch scoring: Not recommended . Online prediction: Not recommended . Although you can use statistics computed using BigQuery      for instance-level batch/online transformations, it isn't easy because      you must maintain a stats store to be populated during training and      used during prediction.  | Batch scoring: N/A —aggregates like these are computed based on      real-time events. Online prediction: Not recommended —you can process training data,      but it results in training-serving skew because you process serving data      using different tools. |
| Dataflow (Apache Beam)                 | Batch scoring: OK —the same transformation implementation is      applied on data during training and batch scoring. Online prediction: OK —if data at serving time comes from      Pub/Sub to be consumed by Dataflow.      Otherwise, results in training-serving skew.                         | Batch scoring: Not recommended . Online predictions: Not recommended . Although you can use statistics computed using Dataflow      for instance-level batch/online transformations, it isn't easy      because you must maintain a stats store to be populated during training      and used during prediction. | Batch scoring: N/A —aggregates like these are computed      based on real-time events. Online prediction: OK —the same Apache Beam transformation is      applied on data during training (batch) and serving (stream).                                             |
| Dataflow (Apache Beam + TFT)           | Batch scoring: OK —the same transformation implementation is      applied to data during training and batch scoring. Online prediction: Recommended —it avoids training-serving skew      and prepares training data up front.                                                                    | Batch scoring: Recommended . Online prediction: Recommended . Both uses are recommended because transformation logic and computed      statistics during training are stored as a TensorFlow      graph that's attached to the exported model for serving.                                                       | Batch scoring: N/A —aggregates like these are computed      based on real-time events. Online prediction: OK —the same Apache Beam transformation is      applied on data during training (batch) and serving (stream).                                             |
| TensorFlow * ( input_fn & serving_fn ) | Batch scoring: Not recommended . Online prediction: Not recommended . For training efficiency in both cases, it's better to prepare the      training data up front.                                                                                                                              | Batch scoring: Not Possible . Online prediction: Not Possible .                                                                                                                                                                                                                                                  | Batch scoring: N/A —aggregates like these are computed      based on real-time events. Online prediction: Not Possible .                                                                                                                                            |

### Which to Use
- If your data is in stored in BigQuery Tables It would be wise to perform as many transformations as possible in SQL, because of its low cost and very very high speed. 
- If the transformation are not possible using BigQuery then Switch to TensorFlow Transform 

- [Reference1](https://cloud.google.com/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_options_summary)
- [Reference2](https://datatonic.com/insights/tensorflow-transform-bigquery-data-transformation/)
