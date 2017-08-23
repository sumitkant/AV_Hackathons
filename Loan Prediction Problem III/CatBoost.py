
# coding: utf-8

# In[56]:

# Load Libaries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[57]:

# Load Training Data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head()


# In[58]:

train.shape


# In[59]:

print (test.shape)
test.head()


# In[60]:

train.index = train['Loan_ID']
test.index = test['Loan_ID']
train = train[train.columns.drop('Loan_ID')]
test = test[test.columns.drop('Loan_ID')]


# In[61]:

test.index


# In[62]:

train.index


# # Categorical Variables

# In[63]:

train['Credit_History'] = train['Credit_History'].astype('object');
train.dtypes


# In[64]:

# CAtegorical Variables
categorical_vars = train.dtypes.loc[train.dtypes == 'object'].index
categorical_vars


# # Continous Variables

# In[65]:

continous_vars = train.dtypes.loc[train.dtypes != 'object'].index
continous_vars


# In[66]:

train[continous_vars].describe()


# # Handling Missing Values

# In[67]:

# Finding Missing values
np.sum(pd.isnull(train))


# ### Computing Mode

# In[68]:

from scipy.stats import mode
mode(train['Gender'].astype(str)).mode[0]


# ### Imputing categorical variables with mode values

# In[69]:

#Impute values 
for var in categorical_vars[np.sum(pd.isnull(train[categorical_vars])).values != 0]:
    train[var].fillna(mode(train[var].astype(str)).mode[0], inplace = True)
    test[var].fillna(mode(test[var].astype(str)).mode[0], inplace = True)


# In[70]:

# Cheking Missing Values
np.sum(pd.isnull(train[categorical_vars]))


# In[71]:

np.sum(pd.isnull(test[categorical_vars[:-1]]))


# ### Imputing Continous Variables with median Values

# In[72]:

np.sum(pd.isnull(train[continous_vars]))


# In[73]:

# values to treat
continous_vars[np.sum(pd.isnull(train[continous_vars])).values != 0]


# In[74]:

for var in continous_vars[np.sum(pd.isnull(train[continous_vars])).values != 0]:
    train[var].fillna(np.median(train[var].dropna()), inplace = True)
    test[var].fillna(np.median(test[var].dropna()), inplace = True)    


# In[75]:

# Checking values
np.sum(pd.isnull(train[continous_vars]))


# In[76]:

np.sum(pd.isnull(test[continous_vars]))


# # CATBOOST

# In[77]:

train.dtypes


# In[78]:

categorical_features_indices = np.where(train.dtypes != np.float)[0]
categorical_features_indices[:-1]


# ## Model Building

# In[79]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Loan_Status'] = le.fit_transform(train['Loan_Status'].astype(str))
train['Loan_Status'].head()
le.classes_


# In[80]:

from sklearn.model_selection import train_test_split
y = train['Loan_Status']
X = train.loc[:,train.columns != 'Loan_Status']
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)


# In[81]:

#importing library and building model
from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=500, 
                         learning_rate=0.01, 
                         depth=6, 
                         l2_leaf_reg=3, 
                         rsm=1, 
                         loss_function='Logloss')

model.fit(X_train, y_train, cat_features=categorical_features_indices[:-1], eval_set=(X_validation, y_validation), plot=True)


# In[82]:

y_pred = model.predict(X_validation)
from sklearn.metrics import accuracy_score
accuracy_score(y_validation, y_pred)


# In[83]:

y_pred


# In[86]:

y_pred_inverse = le.inverse_transform(y_pred.astype(int))
y_pred_inverse


# ## Entire Training Set

# In[87]:

#importing library and building model
from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=500, 
                         learning_rate=0.01, 
                         depth=6, 
                         l2_leaf_reg=3, 
                         rsm=1, 
                         loss_function='Logloss')

model.fit(X, y, cat_features=categorical_features_indices[:-1])


# In[88]:

y_pred = model.predict(test)
y_pred_inverse = le.inverse_transform(y_pred.astype(int))
y_pred_inverse


# In[89]:

ss = pd.read_csv('Sample_Submission.csv')
ss['Loan_ID'] = test.index
ss['Loan_Status'] = y_pred_inverse
ss.to_csv('mysubmission_catboost.csv', sep=',', index= False)

