
# coding: utf-8

# In[169]:

# Load Libaries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[170]:

# Load Training Data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head()


# In[171]:

train.shape


# In[172]:

print (test.shape)
test.head()


# In[173]:

train.index = train['Loan_ID']
test.index = test['Loan_ID']
train = train[train.columns.drop('Loan_ID')]
test = test[test.columns.drop('Loan_ID')]


# In[174]:

test.index


# In[175]:

train.index


# # Categorical Variables

# In[176]:

train['Credit_History'] = train['Credit_History'].astype('object');
train.dtypes


# In[177]:

# CAtegorical Variables
categorical_vars = train.dtypes.loc[train.dtypes == 'object'].index
categorical_vars


# # Continous Variables

# In[178]:

continous_vars = train.dtypes.loc[train.dtypes != 'object'].index
continous_vars


# In[179]:

train[continous_vars].describe()


# # Handling Missing Values

# In[180]:

# Finding Missing values
np.sum(pd.isnull(train))


# ### Computing Mode

# In[181]:

from scipy.stats import mode
mode(train['Gender'].astype(str)).mode[0]


# ### Imputing categorical variables with mode values

# In[182]:

#Impute values 
for var in categorical_vars[np.sum(pd.isnull(train[categorical_vars])).values != 0]:
    train[var].fillna(mode(train[var].astype(str)).mode[0], inplace = True)
    test[var].fillna(mode(test[var].astype(str)).mode[0], inplace = True)


# In[183]:

# Cheking Missing Values
np.sum(pd.isnull(train[categorical_vars]))


# In[184]:

np.sum(pd.isnull(test[categorical_vars[:-1]]))


# ### Imputing Continous Variables with median Values

# In[185]:

np.sum(pd.isnull(train[continous_vars]))


# In[186]:

# values to treat
continous_vars[np.sum(pd.isnull(train[continous_vars])).values != 0]


# In[187]:

for var in continous_vars[np.sum(pd.isnull(train[continous_vars])).values != 0]:
    train[var].fillna(np.median(train[var].dropna()), inplace = True)
    test[var].fillna(np.median(test[var].dropna()), inplace = True)    


# In[188]:

# Checking values
np.sum(pd.isnull(train[continous_vars]))


# In[189]:

np.sum(pd.isnull(test[continous_vars]))


# # Label Encoder

# In[192]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for var in categorical_vars[:-1]:
    train[var] = le.fit_transform(train[var].astype(str))
    test[var] = le.fit_transform(test[var].astype(str))

test.head()


# In[193]:

train.head()


# # Model Building

# In[194]:

y_train = train['Loan_Status']
X_train = train.loc[:,train.columns != 'Loan_Status']


# In[195]:

X_train.head()


# In[196]:

y_train.head()


# In[197]:

X_test = test
X_test.head()


# ## Logistic Regression

# In[198]:

from sklearn.linear_model import LogisticRegression
logitreg = LogisticRegression(C = 0.2, penalty = 'l2').fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print ('Training Accuracy: ', accuracy_score(y_train, logitreg.predict(X_train)))


# #### Predictions

# In[202]:

y_pred = logitreg.predict(X_test)
y_pred


# #### Sample Submission

# In[204]:

ss = pd.read_csv('Sample_Submission.csv')
ss['Loan_ID'] = X_test.index
ss['Loan_Status'] = y_pred


# In[206]:

ss.head()


# In[209]:

ss.to_csv('mysubmission.csv', sep=',', index= False)

