---
title: Data Science Lifecycle
date: 2023-01-12 00:00:00
description: We are here to help you optimise the way you do business and scale your business to the globe using Quality Data, Machine Learning and Automation.
---

### Understanding business requirements
It includes gathering information about the problem from the users and define the problem what is it.Sometimes it can become very tedious as we have to be very careful while gathering the information about the project.  

### Data acquisition
It includes ETL(Extracts Transforms Loads) and most pouplar tool used for this step is SQL(DB,DW,log-files,Hadoop/Spark).Basically this is the step in which we understand where the data is stored and then obatins or loads the data which we required from various sources or files by using mainly sql.

### Data preparation
Now in this step we will have all the data we need whoch we have acquired from the previous step.Now our work is to clean and pre-pocessed the data,so that it can be made useful for making nay ML algo work.Form e.g removing any special character from the text data if any present,handling missing values or NaN values,standardization,featurization.For this we first try to figure what the data is by applying various libraries and tries to print the data and check if there is any incorrect data or the points which is not useful in our model and then after detecting we removes those points or clean the text data.  

### Exploratory Data Analysis
In  this data visualization is done by using various plots.Most commonly used plots are scatter plots,box plot ,heat map,violon plot,bar plot,contour plot,line plot,pdf,cdf,histogram,geo-map,pair-plots,Q-Q plots,t-SNE etc.Basocally by doing we try figure out what are the features that are important and understand the content of the data more accurately and what trend it is following.  

### Modeling, Evaluation & Interpretation
This is the step where we actually apply various ML algorithms like LR,NB,SVM etc and form the model by evaluating on datasets like D_train,D_CV,D_test and tuning hyperparamter to avoid overfitting or underfittig.Applying any model is not easy,first we have to understand that what are algos which will be perfect for the problem and dataset,and then identifying the right performnace metric is aso an imprtant task in modeling.So we have to try the ML algos with different metrics and try to figure out which one is doing best.  

### Communicate results 
Clean & simple : After making the model this step include communicating the results with the stakeholders or users or executives for which we are making this model.The results should be very clean and understandable,because there can be case where persons like users may not know machine learning.This is also one of the very important steps as it includes convicing the manager/users by communicating the results,if this step fails then all the hardwork of making nodel will go in vain,even if you have made the good project but bad communication can fail your success. 

### Deployment
Once we got the approval from the previos step,now it's time the deploy the model which is s software engineering effort. It can be employed sometimes by machine learning engineer or by sofware engineer.

### Real world testing-A/B testing
In this step the model is tested in the real world environment by using A/B testing.Bascially this is the process where we test our model on real world and get the results which will be helpful in the next step.

### Customer buy-in
In this step we try to convince the user in our model by showing the results of the model on real world environment which we have obtained in the previous step. 

### Operations : retrain models, handle failures
This includes identifying that when,how to retrain the models from time to time which  we have deployed already in tyhe real world.And if any failure cases occur in the future and handling that case is also includes in this step.

### Optimization : improve models, more data, more features, optimize code
This is the last step in which we can add more data,more features and improving the models according to the requirement.This is the step which we need to keep doing from time to time according to the requirements and trend in the real world.

