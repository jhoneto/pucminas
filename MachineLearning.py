#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


# In[52]:


df = pd.read_csv('./telcom/telecom_users.csv')
df.head()


# In[53]:


df = df.drop(['Unnamed: 0', 'customerID'], axis = 1)


# In[54]:


df['TotalCharges'] =  df['TotalCharges'].replace(' ', 2298.06)
df['tenure'] =  df['tenure'].replace(0,29)
df['TotalCharges'] =  df['TotalCharges'].astype(float)


# In[55]:


ch = {'Yes': 1, 'No': 0}
df['Churn'] = df['Churn'].map(ch)

df.head()


# In[56]:


X = df.drop('Churn', axis = 1)
y = df['Churn']

num_cols = X.select_dtypes(include = ['int64', 'float64']).columns.to_list()
cat_cols = X.select_dtypes(include = ['object']).columns.to_list()

def label_encoder(df):
    for i in cat_cols:
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
    return df

sc = StandardScaler()
X[num_cols] = sc.fit_transform(X[num_cols])

# Label encoding
X = label_encoder(X)

X.head()


# In[57]:


print(y)


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[59]:


df.head()


# In[60]:


# REGRESSÃO LOGISTICA
lg = LogisticRegression(random_state = 42)
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
y_prob = lg.predict_proba(X_test)[:,1]

# Metricas
print(classification_report(y_test, y_pred))
print("Acuracia", metrics.accuracy_score(y_test, y_pred))

# Matriz de confusão
lg_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (8, 5))
sns.heatmap(lg_cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, 
            annot_kws = {'fontsize': 15}, yticklabels = ['Não cancelou', 'Cancelou'], 
            xticklabels = ['Não cancelaria', 'Cancelaria'])
plt.yticks(rotation = 0)
plt.show()


# In[62]:


# Floresta aletória

rf = RandomForestClassifier(random_state = 22, max_depth = 5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

# Metricas
print(classification_report(y_test, y_pred))
print("Acuracia", metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize = (8, 5))
sns.heatmap(rf_cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, 
            annot_kws = {'fontsize': 15}, yticklabels = ['Não cancelou', 'Cancelou'], 
            xticklabels = ['Não cancelaria', 'Cancelaria'])
plt.yticks(rotation = 0)
plt.show()


# In[ ]:




