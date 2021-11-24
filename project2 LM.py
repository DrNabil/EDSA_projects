#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import csv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[3]:


data1=pd.read_csv("https://raw.githubusercontent.com/DrSaadLa/MLLabs/main/data/housing.csv")


# In[4]:


df=pd.DataFrame(data1)
df


# In[6]:


type(df)
print(dir(df))
print(df.keys())
print(df.shape)
df.info()
df.columns()
df.describe()
df.describe().T


# In[234]:


df.head(5)
df.tail(5)


# In[8]:


## as the var 'address' is an object, so it does not interest me and I will delete :
df1=df.drop('Address',axis=1)


# In[9]:


df1.describe()
df1.describe().T
type(df1)


# In[10]:


plt.style.use('seaborn-whitegrid')
df1.plot.line(x='Price',y=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms']) 
plt.xlabel('Price')
plt.ylabel('Other Variables')
plt.title('Housing Prices',fontdict={'fontweight':'bold','fontsize':22})
plt.legend()
plt.show()


# In[14]:


## to check the correlation:
sns.heatmap(df1.corr(), square=True, cmap='RdYlGn',center=True,annot=True)
## it seems that there is a quite strong correlation between the target variable 'Price' and the features 'Avg. Area Number of Bedrooms' and 'Avg. Area Income' comapring to the other features, so let's plot them:
plt.scatter(x = df1['Avg. Area Number of Bedrooms'], y = df1['Price'])
plt.scatter(x = df1['Avg. Area Income'], y = df1['Price'])
sns.lmplot(x= 'Avg. Area Number of Bedrooms', y = 'Price', data = df1)
sns.lmplot(x= 'Avg. Area Income', y = 'Price', data = df1)
sns.pairplot(df1)


# In[88]:



## according to the scatter plot , the best feature is 'Avg. Area Income'
## we can check its histogram
plt.hist(df1['Avg. Area Income'], bins=50)


# In[80]:


## Simple Linear Regression with SK-Learn
from sklearn.linear_model import LinearRegression


# In[238]:


df_reg = LinearRegression()
## to fit  LinearRegression:
X = df1['Avg. Area Income']
y = df1['Price']
y_reshaped = np.array(y).reshape(-1, 1)
X_reshaped = np.array(X).reshape(-1, 1)
print("Dimensions of y after reshaping: ",y_reshaped.shape)
print("Dimensions of X after reshaping: ", X_reshaped.shape)
df_reg.fit(X=X_reshaped,y=y_reshaped)


# In[239]:


## to check the intercept:
print('The intercept of simple linear regression is', df_reg.intercept_)
## to check the coefficient:
print('The coefficient of simple linear regression is', df_reg.coef_)


# In[240]:


## to save the results we reach in a table:
df1_tab = pd.DataFrame({'Intercept': df_reg.intercept_,
                        'Coef': df_reg.coef_.flatten()})
df1_tab


# In[108]:


## to check the goodness of the linear model(R^2):
df_reg.score(X=X_reshaped,y=y_reshaped)


# In[129]:


## or:
print("The R^2 of model is: {:0.5f}".format(df_reg.score(X_reshaped, y_reshaped)))
# or
print("R^2 (Coefficient of Determination): {:.5f}".format(r2_score(y, y_pred)))


# In[243]:


## Thus, the  feature var 'Avg. Area Income' explains 40% of the changes in the targer var  'Price'
## furthermore, we can check:
1## The mean squared error:
print("Mean squared error(MSE): {:.5f}".format(mean_squared_error(y, y_pred)))
2## The Root Mean squared Error 
print("The Root Mean squared error(RMSE): {:.5f}".format(np.sqrt(mean_squared_error(y, y_pred))))
3## The mean absolute error
print("Mean absolute error(MAE): {:.5f}".format(mean_absolute_error(y, y_pred)))


# In[113]:


## for the Prediction:
y_pred = df_reg.predict(X_reshaped)


# In[244]:


## to plot the result:
plt.scatter(X, y, color='green', alpha=0.5)
plt.plot(X_reshaped, y_pred, color='blue', linewidth=3)
plt.show()


# In[21]:


## to further analyze the linear model, we must validate Train-Test and Model (model Validation):
from sklearn.model_selection import train_test_split
# to split the data:
X_train, X_test, y_train, y_test= train_test_split( X, y,test_size=0.2,random_state=123456)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_test.shape
## standardization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_train.max()
X_train.min()
X_test=scaler.transform(X_test)
# to fit the model on the training set:
df2=LinearRegression()
y_train_reshaped = np.array(y_train).reshape(-1, 1)
X_train_reshaped = np.array(X_train).reshape(-1, 1)
print("Dimensions of y after reshaping: ",y_train_reshaped.shape)
print("Dimensions of X after reshaping: ", X_train_reshaped.shape)
df2.fit(X=X_train_reshaped,y=y_train_reshaped)


# In[246]:


# to predict on train set:
pred_train = df2.predict(X_train_reshaped)
# to predict on test set:
X_test_reshaped = np.array(X_test).reshape(-1, 1)
pred_test = df2.predict(X_test_reshaped)


# In[247]:


## to evaluate the model (model evaluation):
1# The R^2 Score:
print("The R^2 on the train set is: {:.4f}".format(r2_score(pred_train, y_train)))
print("The R^2 on the test set is: {:.4f}".format(r2_score(pred_test, y_test)))
2# The MSE:
print("The MSE on the train set is: {:.4f}".format(mean_squared_error(y_train, pred_train)))
print("The MSE on the test set is: {:.4f}".format(mean_squared_error(y_test, pred_test)))
3# The RMSE:
print("The RMSE on the train set is: {:.4f}".format(np.sqrt(mean_squared_error(y_train, pred_train))))
print("The RMSE on the test set is: {:.4f}".format(np.sqrt(mean_squared_error(y_test, pred_test))))


# In[168]:


## to diagnose the model (Model Diagnostics (Residual Plot)):
# firstly, we calculate the residuals:
resid_train = y_train_reshaped - pred_train
y_test_reshaped = np.array(y_test).reshape(-1, 1)
resid_test = y_test_reshaped-pred_test
# Then, we scatter-plot the training data:
plt.figure(figsize= (12, 6))
train = plt.scatter(x = pred_train, y = resid_train , c = 'b', alpha=0.5)

# also, we make the same for the testing data:
test = plt.scatter(pred_test, resid_test , c = 'r', alpha=0.5)

# we plot a horizontal axis line at 0
plt.hlines(y = 0, xmin = -10, xmax = 50)

# we acn Label them:
plt.legend((train, test), ('Training','Test'), loc='upper left')
plt.title('Residual Plots')
plt.show()


# In[183]:


## now, we can validate deeply the linear model (Linear Regression with Cross Validation):
from sklearn.model_selection import KFold
# first, we create KFold generator:
kf = KFold(n_splits=5, shuffle=False)
# we create splits:
splits = kf.split(X)
# we print the number of indices:
for train_index, val_index in splits:
    print(len(train_index), len(val_index))
# we print the indexes of one train_index and one val_index:
print(train_index, val_index)
## or:
for train_index, val_index in splits:
    print("Number of training indices: {}".format(len(train_index)))
    print("Number of validation indices: {}".format(len(val_index)))
    print()
          


# In[248]:


## to fit the model:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
df2 = LinearRegression()
# first, we ceate KFold generator:
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
## we create splits
splits = kf.split(X)
# we access the training and validation indices of splits by using a for loop:
for train_index, val_index in splits:
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
# to fit a linear regression:    
X_train1=np.array(X_train).reshape(-1, 1)
y_train1=np.array(y_train).reshape(-1, 1)
df2.fit(X_train1, y_train1)


# In[249]:


# to make predictions, and print MSE and MAE:
X_val_reshaped=np.array(X_val).reshape(-1, 1)
preds = df2.predict(X_val_reshaped)
print("The MSE metric: {:.5f}".format(mean_squared_error(y_val, preds)))
print("The MAE metric: {:.5f}".format(mean_absolute_error(y_val, preds)))
print()


# In[263]:


## to reach Automating K-fold Cross Validation with cross_val_score() Function, we follow  the following steps:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
# To perform cross validation,we perform cross validation:
df2= LinearRegression()
y_reshaped = np.array(y_train).reshape(-1, 1)
X_reshaped = np.array(X_train).reshape(-1, 1)
print("Dimensions of y after reshaping: ",y_train_reshaped.shape)
print("Dimensions of X after reshaping: ", X_train_reshaped.shape)
cv_res= cross_val_score(df2, X_reshaped, y_reshaped, cv = 10)


# In[265]:


print(type(cv_res))


# In[264]:


cv_res


# In[267]:


# to compute the average R^2:
avg_r_sq = np.mean(cv_res)
print(f"The average R^2 is: {avg_r_sq :.5f}")


# In[268]:


## to score the cross validation results with different metrics:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer


# In[279]:


# to make mae scorer:
MAE = make_scorer(mean_absolute_error)
# to perform 10-fold CV
cv_results = cross_val_score(df2,X_reshaped, y_reshaped,cv = 10,scoring=MAE)


# In[285]:


print(cv_results)
print(cv_results.mean())
MSE = make_scorer(mean_squared_error)
# perform 10-fold CV
cv_results= cross_val_score(df2,X_reshaped, y_reshaped,cv = 10,scoring=MSE)
print(cv_results)
print(cv_results.mean())


# In[286]:


## another options to deal with cross validation
k_10_cv = cross_val_score(df2, X_reshaped, y_reshaped, cv = 10)
print(np.mean(k_10_cv ))


# In[289]:


get_ipython().run_cell_magic('timeit', '', 'cross_val_score(df2, X_reshaped, y_reshaped, cv = 10)')


# In[291]:


# import train_test_split
from sklearn.model_selection import train_test_split 
# to create training and temporay sets:
X_temp, X_test, y_temp, y_test=train_test_split(X, y,test_size= 0.2,random_state=12345)
# Create validation and train sets
X_train, X_val, y_train, y_val  =train_test_split(X_temp, y_temp,test_size= 0.2,random_state=12345)
print(X_train.shape, X_test.shape, X_val.shape, y_train.shape,y_test.shape, y_val.shape)


# In[ ]:




