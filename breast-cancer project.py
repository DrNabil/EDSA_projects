#!/usr/bin/env python
# coding: utf-8

# In[206]:


import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; import seaborn as sns
from numpy import *
from matplotlib.pyplot import *
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV


# In[207]:


data=pd.read_csv("https://raw.githubusercontent.com/DrSaadLa/PythonTuts/main/TreeBasedModels/breastcancer.csv")
data.info()
print("*"*50)
data.describe()
print("*"*50)
data.head()
print("*"*50)
data.head(10)
print("*"*50)
data.shape
data['diagnosis'].unique()
data['diagnosis'].value_counts()
data.isnull().sum()
print("*"*50)


# In[211]:


data['diagnosis'] = data['diagnosis'].apply(lambda x:1 if x=='M' else 0)
data['diagnosis'].unique()


# In[212]:


data.corr().style.background_gradient(cmap='PuBu')


# In[230]:


X = data.drop('diagnosis', axis=1)
y = data['diagnosis'].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2,stratify=y,random_state= 1)
X_train.head().T


# ## Check the model using Random Forest

# In[216]:


data_reg = RandomForestRegressor(random_state= 1)
data_reg.fit(X_train, y_train)
preds = data_reg.predict(X_test)
rmse = np.sqrt(MSE(preds, y_test))
print("Test set RMSE: {:.5f}".format(rmse))


# In[217]:


## Plotting The Variable Importance

from pandas import Series
pd.options.display.float_format = '{:,.3f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')


# In[218]:


features_importances = pd.Series(data=data_reg.feature_importances_,
                        index= X_train.columns)
importances_sorted = features_importances.sort_values()
importances_sorted.plot(color='lightblue',
                        kind='barh')
plt.title('Features Importances')
plt.show()


# In[219]:


data_reg = RandomForestRegressor(random_state= 1)
params_data_reg = {
    'n_estimators': [100, 350, 500, 1000], 
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1, 20, 30, 50]
}
grid_data_reg = GridSearchCV(estimator=data_reg,
                       param_grid= params_data_reg,
                       scoring='neg_mean_squared_error',
                       cv=8,
                       verbose=1,
                       n_jobs=-1)
grid_data_reg.fit(X_train, y_train)
best_model = grid_data_reg.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = np.sqrt(MSE(y_test, y_pred))
print('Test RMSE of best model: {:.5f}'.format(rmse_test)) 


# ## Check the model using Ensemble Algorithm

# In[221]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler


# In[237]:


scaler=StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
data_logreg = LogisticRegression(random_state=1)
data_knn = KNN(n_neighbors=30)
data_dt = DecisionTreeClassifier(min_samples_leaf= 0.13, 
                            random_state=1)
classifiers = [('Logistic Regression', data_logreg), 
               ('K Nearest Neighbours', data_knn), 
               ('Classification Tree', data_dt)]
for clf_name, clf in classifiers:    
    clf.fit(X_train1, y_train)    
    y_pred1 = clf.predict(X_test1)  
    accuracy = accuracy_score(y_pred1, y_test) 
    print('{:20}: {:.5f}'.format(clf_name,  accuracy))


# In[238]:


data_vc = VotingClassifier(estimators=classifiers)     
data_vc.fit(X_train1, y_train)   
y_pred1 = data_vc.predict(X_test1)
accuracy = accuracy_score(y_pred1, y_test)
print('Voting Classifier: {:.5f}'.format(accuracy))


# ## Check the model using Bagging

# In[229]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor


# In[232]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)                                                             


# In[243]:


data_dtree = DecisionTreeClassifier(random_state=1)
data_bagging = BaggingClassifier(base_estimator=data_dtree,n_estimators= 500,random_state=1)
data_bagging.fit(X_train, y_train)
y_pred = data_bagging.predict(X_test)
acc_test = accuracy_score(y_pred, y_test)
print('Test set accuracy of bc: {:.5f}'.format(acc_test)) 


# In[246]:


data_dtree1 = DecisionTreeClassifier(min_samples_leaf=8,random_state=1)
data_bc = BaggingClassifier(base_estimator= data_dtree1, n_estimators= 500 ,oob_score=True,random_state=1, n_jobs=-1,verbose=1)
data_bc.fit(X_train, y_train)
y_pred1 = data_bc.predict(X_test)
test_acc = accuracy_score(y_pred, y_test)
oob_acc = data_bc.oob_score_
print("*"*50)
print('Test set accuracy: {:.5f}'.format(test_acc))
print("*"*50)
print('The OOB accuracy: {:.5f}'.format(oob_acc))


# In[ ]:




