#!/usr/bin/env python
# coding: utf-8

# # HOUSING PRICE PREDICTION

# ## 1. Import Required Libraries and Read Dataset

# In[1]:


import numpy as np
import pandas as pd
import os
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
data.head()


# ## 2. Data Exploration

# In[2]:


data


# In[3]:


print(np.shape(data))# Dimension of the dataset


# In[4]:


# Summarize the data to see the distribution of data
print(data.describe())


# In[5]:


data.info() # Concise summary of a DataFrame


# In[6]:


data.isnull().sum() #cheaking null value in the data frame


# ## 3. Data Visualization and Analysis

# #### Box plot with Hyper parameter tuning

# In[7]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col,value in data.items():
    sns.boxplot(y=col, data=data, ax=ax[index])
    index +=1
    
#hyper parameter tuning
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# #### Hist Plot with Hyper parameter tuning

# In[8]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col,value in data.items():
    sns.histplot(value, kde=True, ax=ax[index])
    index +=1
#hyper parameter tuning
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# #### Max-Min Normalization

# In[9]:


cols = ['CRIM','ZN','TAX','B'] 
for col in cols:
    #finding minimum and maximum of that columns
    minimum = min(data[col])
    maximum = max(data[col])
    data[col]=(data[col]-minimum)/(maximum - minimum)


# In[10]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col,value in data.items():
    sns.histplot(value, kde=True, ax=ax[index])
    index +=1
#hyper parameter tuning
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# #### Standardize the Data

# In[11]:


from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

#fit out data
scaled_cols = scalar.fit_transform(data[cols])
scaled_cols = pd.DataFrame(scaled_cols,columns =cols)
scaled_cols.head()


# In[12]:


for col in cols:
    data[col]=scaled_cols[col]


# In[13]:


#hist plot
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col,value in data.items():
    sns.histplot(value, kde=True, ax=ax[index])
    index +=1
#hyper parameter tuning
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# #### Correlation Matrix

# In[14]:


corr = data.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap="coolwarm")


# In[15]:


sns.regplot(y=data['MEDV'],x=data['LSTAT']) # scatter plot correlation between MEDV and LSTAT


# In[16]:


sns.regplot(y=data['MEDV'],x=data['RM'])# scatter plot correlation between MEDV and RM


# ## 4. Splittng the Data

# In[17]:


#train and test split input split
from sklearn.model_selection import cross_val_score, train_test_split
X = data.drop(columns = ['MEDV','RAD'],axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=2)

# shape of spiltted data
print("The shape of X_train :",X_train.shape)
print("The shape ofX_test :",X_test.shape)
print("The shape of y_train :",y_train.shape)
print("The shape of y_test :",y_test.shape)


# ## 5. Models Building and Testing

# #### Linear Regression

# In[18]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#model training
def train(model,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=2)
    model.fit(X_train,y_train)
    
    
    #prediction of the model
    pred = model.predict(X_test)
    
    #perform cross validation
    cv_score = cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    # Calculate accuracy score (R^2 score)
    r2 = r2_score(y_test, pred)
    
    print("model report: ")
    print("MSE:",mean_squared_error(y_test,pred))
    print("CV Score",cv_score)
    print("R^2 Score:", r2)


# In[19]:


# Create the pipeline with StandardScaler and LinearRegression
model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())

# Train the model
train(model, X, y)

# Extract the linear regression model from the pipeline
linear_model = model.named_steps['linearregression']
coef = pd.Series(linear_model.coef_, X.columns).sort_values()

# Plot the model coefficients
coef.plot(kind='bar', title='Model Coefficients')
plt.show()


# #### Decision Tree model

# In[20]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor 
dtr = DecisionTreeRegressor(max_depth=5)

# Fit the model on Training dataset
dtr.fit(X_train,y_train)


# In[21]:


# Predictions of  decision Tree Regressor on Testing Data
y_pred_dtr=dtr.predict(X_test)

# Accuracy Score of Model
from sklearn.metrics import mean_absolute_percentage_error
error = mean_absolute_percentage_error(y_pred_dtr,y_test)

print("Accuracy of Decision Tree Regressor is :%.2f "%((1 - error)*100),'%')

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dtr, edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Decision Tree Regressor: Actual vs Predicted Prices')
plt.show()

# Additional metrics
mse = mean_squared_error(y_test, y_pred_dtr)
r2 = r2_score(y_test, y_pred_dtr)
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)


# #### Ridge Regression Model

# In[22]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=3.8, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)


# In[23]:


from sklearn.metrics import r2_score

# Predicting Cross Validation Score the Test set results
cv_ridge = cross_val_score(estimator = ridge_pipe, X = X_train, y = y_train.ravel(), cv = 10)

# Predicting R2 Score the Test set results
y_pred_ridge_train = ridge_pipe.predict(X_train)
r2_score_ridge_train = r2_score(y_train, y_pred_ridge_train)

# Predicting R2 Score the Test set results
y_pred_ridge_test = ridge_pipe.predict(X_test)
r2_score_ridge_test = r2_score(y_test, y_pred_ridge_test)

# Predicting RMSE the Test set results
rmse_ridge = (np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)))
print('CV: ', cv_ridge.mean())
print('R2_score (train): ', r2_score_ridge_train)
print('R2_score (test): ', r2_score_ridge_test)
print("RMSE: ", rmse_ridge)

# Plot actual vs predicted prices for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge_test, edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Ridge Regression: Actual vs Predicted Prices')
plt.show()


# #### Random Forest model

# In[24]:


# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth = 10, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)

# Fit the model on Training datset
rfr.fit(X_train,y_train)


# In[25]:


# Predictions of  Ranforest Forest Regressor on Testing Data
y_pred_rfr = rfr.predict(X_test)

# Accuracy Score of Model
error = mean_absolute_percentage_error(y_pred_rfr,y_test)
print("Accuracy of Random Forest Regressor is :%.2f "%((1 - error)*100),'%')

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rfr, edgecolor='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Regressor: Actual vs Predicted Prices')
plt.show()

# Additional metrics
mse = mean_squared_error(y_test, y_pred_rfr)
r2 = r2_score(y_test, y_pred_rfr)
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)


# ### BEST MODEL
# #### Ridge Regression Model with Accuracy - 91% 

#  

# ## Thank You
