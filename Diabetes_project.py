#!/usr/bin/env python
# coding: utf-8

# Importing the Libraries

# In[43]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Data Collection and Analysis

# Diabetes Dataset

# In[44]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv("pima-indians-diabetes.csv") 


# In[45]:


diabetes_dataset


# In[46]:


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[47]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[48]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[49]:


diabetes_dataset['Outcome'].value_counts()


# In[50]:


diabetes_dataset.groupby('Outcome').mean()


# In[51]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[52]:


print(X)


# In[53]:


print(Y)


# Data Standardization

# In[54]:


scaler = StandardScaler()


# In[55]:


scaler.fit(X)


# In[56]:


standardized_data = scaler.transform(X)


# In[57]:


print(standardized_data)


# In[58]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[59]:


print(X)
print(Y)


# Train Test Split

# In[60]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[61]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model using SVM

# In[62]:


classifier = svm.SVC(kernel='linear')


# In[63]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# Accuracy Score using SVM

# In[64]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[65]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[66]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[67]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Training the Model using Random Forest Classifier

# In[68]:


num_trees = 100
max_features = 3


# In[69]:


clf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[70]:


clf.fit(X_train,Y_train)


# Accuracy Score using RandomForestClassifier

# In[71]:


Y_pred=clf.predict(X_test)


# In[78]:


print("Accuracy score of the test data is :",metrics.accuracy_score(Y_test, Y_pred))


# Making a Predictive System

# In[84]:


input_data = (6,148,72,35,0,33.6,0.627,50)


# In[85]:


# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[86]:


# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[87]:


# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)


# In[88]:


prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic.')
else:
  print('The person is diabetic.')


# In[ ]:





# In[ ]:




