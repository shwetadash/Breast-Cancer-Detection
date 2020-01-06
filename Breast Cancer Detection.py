#!/usr/bin/env python
# coding: utf-8

# In[9]:


#checking python version of libraries
import sys
import numpy
import scipy
import pandas
import matplotlib
import sklearn

print('Python:{}'.format(sys.version))
print('scipy:{}'.format(scipy.__version__))
print('pandas:{}'.format(pandas.__version__))
print('numpy:{}'.format(numpy.__version__))
print('matplotlib:{}'.format(matplotlib.__version__))
print('sklearn:{}'.format(sklearn.__version__))


# In[3]:


import numpy as np #for mathematical calculations
from sklearn import preprocessing,cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


# In[13]:


#load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"


# In[14]:



names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)


# In[15]:


df.head()


# In[16]:


#if there is no data df.replace gives a question mark in the place
#So we need to fill the ? with some value like -999999 and tell python to ignore the data
df.replace('?',-99999,inplace=True)

#inplace=true makes changes to the original dataset whereas if you set it as false it just creates a copy of the dataset 
#and makes changes to it

print(df.axes)
#we notice that there are 699 different data point and we see the column names
df.drop(['id'],1,inplace=True)
#we will drop the 1st column that is the id column because it wont be used to do the machine learning program


# In[17]:


print(df.shape)


# In[18]:


#printing the summary statistics of the data
df.describe()


# In[19]:


df.head()


# In[22]:


df['class'].values
#so there are 2 values of classes :2 and 4


# Here 2 represents benign tumor and 4 represents malignant tumor

# In[28]:


#plotting histogram for each variable
df.hist(figsize=(15,15))
plt.show()


# In[30]:


#creating a scatter plot matrix
scatter_matrix(df,figsize=(18,18))
plt.show()


# In[33]:


#Next we have to split the data into training set and testing set and keep rest of the data for validation
#Creating X and Y datasets
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])


# In[34]:


X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)


# In[37]:


seed=8
scoring='accuracy'


# In[38]:


# Define models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC())) #support vector machine

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[39]:


# Make predictions on validation dataset

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


# In[40]:


clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)


# In[ ]:




