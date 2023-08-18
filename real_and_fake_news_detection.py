# -*- coding: utf-8 -*-




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Importing Modeling Algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

# Importing Modeling Helpers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics


# Computations
import itertools

# Visualization
import matplotlib.pyplot as plt

test  = pd.read_csv ("test.csv")
submit  = pd.read_csv ("submit.csv")

train = pd.read_csv("test.csv")

train.head()

# Displaying the shape of the training, test, and submission datasets
print("Shape of Training Data:", train.shape)
print("Shape of Test Data:", test.shape)
print("Shape of Submission Data:", submit.shape)


train.info()

train.isnull().sum()

train.dtypes.value_counts()

test=test.fillna(' ')
train=train.fillna(' ')

# Create a column with all the data available
test['total']=test['title']+' '+test['author']+' '+test['text']
train['total']=train['title']+' '+train['author']+' '+train['text']

# Have a glance at our training set
train.info()
train.head()





