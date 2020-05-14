#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import statsmodels.tools as sm
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
import scikitplot as skplt 
import warnings
warnings.filterwarnings('ignore')


# In[68]:


url="https://raw.githubusercontent.com/aniharutyunyan1/Bus1234/master/CRA_client_data%202.csv"
data=pd.read_csv(url)


# In[21]:


data.info()


# In[23]:


data.describe()


# In[22]:


data.head()


# In[27]:


print("Duplicates:", data.duplicated().sum())
print("Missing values:", data.isna().sum().sum())
print("Single valued columns:", data.columns[data.nunique()==1])


# In[30]:


data.isnull().any()


# In[36]:


categorical_colnames=data.select_dtypes(include="object").columns
for colname in categorical_colnames:
    val_count=data[colname].value_counts()
    print(colname,"\n",val_count,"\n","\n")
    print("--------------------")


# In[39]:


drops = ["SMARTV40_APR14","CLIENT_ID","SEGMENT_CODE","KEYID","XFC06","XFC07","DECILE","DEM10"]
data = data.drop(drops,axis=1)


# In[40]:


plt.figure(figsize=(9,5))
sns.countplot(data.RESPONSE_FLAG)
plt.title("RESPONSE")
plt.show()


# In[ ]:


co


# In[30]:


data.RESPONSE_FLAG.value_counts(normalize=True).mul(100).rename("RESPONSE FLAG percentage")


# In[41]:


print("Duplicates:", data.duplicated().sum())
print("Missing values:", data.isna().sum().sum())
print("Single valued columns:", data.columns[data.nunique()==1])


# In[ ]:


numeric_colnames=data.dtypes[data.dtypes=="O"].index.tolist()
rates_list=[]
for colname in numeric_colnames:
    rates=data.groupby(colname)["Attrition"].value_counts(normalize=True).rename("percentage").mul(100).reset_index()
    rates_list.append(rates)


# In[ ]:


rates_list[0]


# In[43]:


data=pd.get_dummies(data,drop_first=True)

X=data.drop('RESPONSE_FLAG', axis=1)
Y=data.RESPONSE_FLAG


# In[73]:


X=sm.add_constant(X)


# In[44]:


X0, X1, Y0, Y1=train_test_split(X,Y, test_size=0.25, random_state=42)


# In[72]:


train=train.fillna(train.median())


# In[ ]:


def make_tree(model):
    tree_0 = model
    tree_0.fit(X0,Y0)


# In[ ]:


Y0_0_proba = tree_0.predict_proba(X0)[:,1] 
Y1_0_proba = tree_0.predict_proba(X1)[:,1] 

Y0_0_class = np.where(Y0_0_proba>0.5,1,0) 
Y1_0_class = np.where(Y1_0_proba>0.5,1,0) 

print("Train ROC AUC by tree_0:",roc_auc_score(Y0,Y0_0_proba))
print("Test ROC AUC by tree_0:",roc_auc_score(Y1,Y1_0_proba))

print("\n")

print("Train Recall by tree_0:",recall_score(Y0,Y0_0_class))
print("Test Recall by tree_0:",recall_score(Y1,Y1_0_class))


# In[ ]:


make_tree(DecisionTreeClassifier(random_state=42)


# In[ ]:


#change the parameters
make_tree(DecisionTreeClassifier(max_depth=7,class_weight="balanced",random_state=42))


# In[ ]:


make_tree(DecisionTreeClassifier(min_samples_leaf=85,random_state=42))


# In[ ]:


make_tree(DecisionTreeClassifier(max_depth=7,min_samples_leaf=85,random_state=42))


# In[ ]:


#BEST Hyperparameters
tree_best = DecisionTreeClassifier(max_depth=7,min_samples_leaf=85,random_state=42)
tree_best.fit(X0,Y0)


# In[ ]:


pd.DataFrame(data=tree_best.feature_importances_,index=X0.columns)


# In[ ]:


tree_final = DecisionTreeClassifier(max_depth=7,min_samples_leaf=85,random_state=42)
tree_final.fit(X0[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company"]],Y0)

Y0_best_proba = tree_final.predict_proba(X0[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company"]])[:,1]
Y1_best_proba = tree_final.predict_proba(X1[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company"]])[:,1]

Y0_best_class = np.where(Y0_best_proba>0.5,1,0)
Y1_best_class = np.where(Y1_best_proba>0.5,1,0)

print("Train ROCAUC:",roc_auc_score(Y0,Y0_best_proba))
print("Test ROCAUC:",roc_auc_score(Y1,Y1_best_proba))


# In[ ]:


plt.figure(figsize=(30,30))
plot_tree(tree_final,filled=True,feature_names=X0.columns)
plt.show()


# In[ ]:





# In[ ]:




