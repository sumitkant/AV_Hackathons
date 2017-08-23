
# coding: utf-8

# # BIG MART SALES
# ### Problem Statement
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to **build a predictive model and find out the sales of each product at a particular store.**
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
#  
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

# In[1]:

# Load Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Load Training Data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
train.head()


# In[3]:

train.columns


# In[4]:

train.shape


# In[5]:

# Number of Products
len(train.Item_Identifier.unique())


# In[6]:

# Number of Stores
len(train.Outlet_Identifier.unique())


# In[7]:

# Create ITEM_WEIGHT TABLE
items = train.Item_Identifier.unique()
item_weight = []
for item in items:
    w = train.Item_Weight[train.Item_Identifier == item].dropna().values
    if len(w) == 0:
        x = 0
    else:
        x = w[0].astype(float)
    item_weight = np.append(item_weight, x)


# In[8]:

iw = pd.Series(item_weight, index = items)
iw.head()


# In[9]:

# Removing Missing values in weight..
# For weights which did not exist  were substituted with '0'
for item in iw.index:
    train.Item_Weight[train.Item_Identifier == item] = iw[item]


# In[29]:

# Removing Missing values in weight..
# For weights which did not exist  were substituted with '0'
for item in iw.index:
    test.Item_Weight[test.Item_Identifier == item] = iw[item]


# In[10]:

# Create ITEM_WEIGHT TABLE
outlets = train.Outlet_Identifier.unique()
outlet_size = []
for outlet in outlets:
    s = train.Outlet_Size[train.Outlet_Identifier == outlet].dropna().values
    if len(s) == 0:
        s=0
    else:
        s = s[0]
    outlet_size = np.append(outlet_size, s)


# In[11]:

os = pd.Series(outlet_size, index = outlets)
os


# In[12]:

pd.crosstab(train.Outlet_Size, train.Outlet_Type)


# ### Groccery Stores have a small size

# In[13]:

x = []
for out in os.index:
    x =  np.append(x,train.Outlet_Type[train.Outlet_Identifier == out].unique())
size_type = pd.DataFrame({'size': os.values,'type':x}, index = os.index)
size_type


# In[14]:

train.Outlet_Size[train.Outlet_Type == 'Grocery Store'] = 'Small'


# In[25]:

test.Outlet_Size[test.Outlet_Type == 'Grocery Store'] = 'Small'


# In[15]:

# Missing Values Left
np.sum(pd.isnull(train.Outlet_Size))


# ### Supermarket Type1 in Tier 2 is small in size

# In[16]:

pd.crosstab(train.Outlet_Size, [train.Outlet_Type,train.Outlet_Location_Type])


# In[17]:

train.Outlet_Size[(train.Outlet_Type == 'Supermarket Type1') & (train.Outlet_Location_Type == 'Tier 2')] = 'Small'


# In[23]:

test.Outlet_Size[(test.Outlet_Type == 'Supermarket Type1') & (test.Outlet_Location_Type == 'Tier 2')] = 'Small'


# ### Checking missing values

# In[31]:

# Missing Values Left in the training set
np.sum(pd.isnull(train))


# In[30]:

# Missing Values Left in test set
np.sum(pd.isnull(test))


# ### Reindexing Test and Training sets

# In[108]:

train_indexed = train.set_index(['Item_Identifier', 'Outlet_Identifier'],drop = True)
train_indexed.head()


# In[109]:

# Combinations of product and items
len(train_indexed.index.unique())


# In[110]:

test_indexed = test.set_index(['Item_Identifier', 'Outlet_Identifier'],drop = True)
test_indexed.head()


# # Vairables

# In[111]:

train_indexed.dtypes


# ### Item Variables

# In[112]:

var_items = train_indexed.dtypes[:-5].index
var_items
# var_outlets = train_indexed.dtypes[5:-1]


# #### Outlet Variables

# In[113]:

var_outlets = train_indexed.dtypes[5:-1].index
var_outlets
# var_outlets = train_indexed.dtypes[5:-1]


# # Label Encoding

# In[114]:

train_indexed.head()


# In[115]:

# Categorical Variables
cat_vars = train_indexed.columns[train_indexed.dtypes == 'object']
cat_vars


# In[138]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for var in cat_vars:
    train_indexed[var] = le.fit_transform(train_indexed[var].astype(str))
    test_indexed[var] = le.fit_transform(test_indexed[var].astype(str))
train_indexed.head()


# In[139]:

test_indexed.head()


# # Model Building

# In[117]:

y = train_indexed['Item_Outlet_Sales']
X = train_indexed[train_indexed.columns[:-1]]
X.columns


# In[118]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[119]:

from sklearn.ensemble import RandomForestRegressor
rmse_arr = []
for i in np.arange(1,300,10):
    rf = RandomForestRegressor(n_estimators=i, 
                           criterion='mse', 
                           max_depth=10, 
                           min_samples_leaf=30).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse_arr = np.append(rmse_arr, np.sqrt(np.sum((y_pred - y_test)**2)/len(y_pred)))


# In[120]:

plt.figure()
plt.plot(np.arange(1,300,10), rmse_arr)


# In[123]:

from sklearn.ensemble import RandomForestRegressor
rmse_arr_d = []
x = np.arange(1,20)
for i in x:
    rf = RandomForestRegressor(n_estimators=70, 
                           criterion='mse', 
                           max_depth=i, 
                           min_samples_leaf=30).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse_arr_d = np.append(rmse_arr_d, np.sqrt(np.sum((y_pred - y_test)**2)/len(y_pred)))


# In[124]:

plt.figure()
plt.plot(x, rmse_arr_d)


# In[132]:

from sklearn.ensemble import RandomForestRegressor
rmse_arr_l = []
x = np.arange(1,100,3)
for i in x:
    rf = RandomForestRegressor(n_estimators=70, 
                           criterion='mse', 
                           max_depth=6, 
                           min_samples_leaf=i).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse_arr_l = np.append(rmse_arr_l, np.sqrt(np.sum((y_pred - y_test)**2)/len(y_pred)))


# In[133]:

plt.figure()
plt.plot(x, rmse_arr_l,'-o')


# ### Random Forest with  optimum values
# 

# In[134]:

rf = RandomForestRegressor(n_estimators=70, 
                       criterion='mse', 
                       max_depth=6, 
                       min_samples_leaf=22).fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse = np.sqrt(np.sum((y_pred - y_test)**2)/len(y_pred))
print ('RMSE on the Cross validation Set is: ', rmse)


# # Modelling on the entire test set

# In[140]:

X_train = X
y_train = y
X_test = test_indexed


# In[141]:

X_test.head()


# In[142]:

rf = RandomForestRegressor(n_estimators=70, 
                       criterion='mse', 
                       max_depth=6, 
                       min_samples_leaf=22).fit(X_train, y_train)
y_pred = rf.predict(X_test)


# ## Submimssion

# In[144]:

ss = pd.read_csv('ss.csv')
ss.head()


# In[150]:

ss.Item_Outlet_Sales = y_pred
ss.head()


# In[151]:

ss.to_csv('rf_sub_sumitkant.csv', sep=',', index= False)


# In[ ]:



