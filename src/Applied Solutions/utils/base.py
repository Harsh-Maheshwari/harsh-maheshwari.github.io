# Importing css
from IPython.core.display import HTML
css = lambda : HTML(open("Assets/css/custom.css", "r").read())

import warnings
warnings.filterwarnings("ignore")

## base packages
import math
import re
import time
from collections import Counter, defaultdict
import six
import sys
from itertools import chain 

# memory management
import gc 

## reading data
import pandas as pd

## mathematics & scientific computing
import numpy as np
from scipy import stats
from scipy.sparse import hstack
from scipy.stats import norm

## distributions
from numpy.random import default_rng

## eda
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns

## nlp
import nltk
from nltk.corpus import stopwords

## reports
from dataprep.eda import create_report

## imbalance data
from imblearn.over_sampling import SMOTE

# Outlier Detection
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF


## models
# from mlxtend.classifier import StackingClassifier
# import lightgbm as lgb
# import xgboost as xgb


## sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

### feature engineering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

### models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

### model tuning
from sklearn.calibration import CalibratedClassifierCV

### model analysis
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score

### metrics
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, normalized_mutual_info_score, mean_squared_error, classification_report

sys.modules['sklearn.externals.six'] = six
sns.set(style='white', context='notebook', palette='deep') 


def glimpse(df):
    num_nas = df.isnull().sum()
    num_of_rows = df.shape[0]
    num_nas_ratio = num_nas/num_of_rows*100
    dtypes = df.dtypes
    num_uniques = df.nunique()
    num_uniques_ratio = num_uniques/num_of_rows*100
    info = pd.concat([dtypes, num_nas, num_nas_ratio, num_uniques, num_uniques_ratio], axis=1)
    info.columns =['dtype', '# of Nas', '% of Na', '# of uniques', '% of unique']
    info = info.T
    info.columns.name = 'Info'
    sample = df.sample(10).copy()
    sample.columns.name = 'Sample'
    stats = df.describe()
    stats.columns.name = 'Stats'

    print('Shape: ', df.shape)
    display(info)
    display(df.dtypes.value_counts().to_frame('Count of dtypes'))
    display(stats)
    display(sample)




def plot_confusion_matrix(test_y, predict_y):
    """ 
    This function plots the confusion matrices given y_i, y_i_hat.
    """
    
    C = confusion_matrix(test_y, predict_y)    
    # divid each element of the confusion matrix with the sum of elements in that column
    A =(((C.T)/(C.sum(axis=1))).T)    
    # divid each element of the confusion matrix with the sum of elements in that row
    B =(C/C.sum(axis=0))    
    labels = [1,2,3,4,5,6,7,8,9]
    
    fig, ax = plt.subplots(3,1, figsize = (16,16))
    
    # representing A in heatmap format
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_xlabel('Predicted Class')
    ax[0].set_ylabel('Original Class')
    ax[0].set_title("Confusion matrix")

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels, ax=ax[1])
    ax[1].set_xlabel('Predicted Class')
    ax[1].set_ylabel('Original Class')
    ax[1].set_title("Precision matrix (Columm Sum=1)")

    
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels, ax=ax[2])
    ax[2].set_xlabel('Predicted Class')
    ax[2].set_ylabel('Original Class')
    ax[2].set_title("Recall matrix (Row sum=1)")
    plt.tight_layout()


def get_response_coded_feature(train_df, cv_df, test_df, feature, target, laplace_alpha):
    """
    get_response_coded_feature for high dimensionality Categorical features based on the no of class labels
    TODO: Check if we can separate this function for train, cv, and test. Currenlty it used train information and encodes all 3 sets based on that 

    """
    value_count = train_df[feature].value_counts()
    unique_target_classes = sorted(train_df[target].unique())
    no_of_unique_target_classes = len(unique_target_classes)
    numerator = train_df.groupby([feature, target]).size()
    numerator = numerator.unstack(fill_value=0) + laplace_alpha
    denominator = train_df.groupby([feature]).size() + laplace_alpha*no_of_unique_target_classes
    dict_of_coded_classes = (numerator.stack()/denominator).unstack().T.to_dict('list')  
    constant_base = 1/no_of_unique_target_classes
    uniques_in_train = list(value_count.index)

    response_coded_column_train = []
    for index, row in train_df.iterrows():
        if row[feature] in uniques_in_train:
            response_coded_column_train.append(dict_of_coded_classes[row[feature]])
        else:
            response_coded_column_train.append([constant_base]*no_of_unique_target_classes)

    response_coded_column_cv = []
    for index, row in cv_df.iterrows():
        if row[feature] in uniques_in_train:
            response_coded_column_cv.append(dict_of_coded_classes[row[feature]])
        else:
            constant_base = 1/no_of_unique_target_classes
            response_coded_column_cv.append([constant_base]*no_of_unique_target_classes)

    response_coded_column_test = []
    for index, row in test_df.iterrows():
        if row[feature] in uniques_in_train:
            response_coded_column_test.append(dict_of_coded_classes[row[feature]])
        else:
            constant_base = 1/no_of_unique_target_classes
            response_coded_column_test.append([constant_base]*no_of_unique_target_classes)
            
    return np.array(response_coded_column_train), np.array(response_coded_column_cv), np.array(response_coded_column_test)
    


