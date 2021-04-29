#!/usr/bin/env python
# coding: utf-8



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

df = pd.read_csv('./telcom/telecom_users.csv')
df.head()
df.shape
df.isnull().values.any()
cat_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'Churn']
num_columns = ['TotalCharges','MonthlyCharges','tenure']

for col in df[cat_columns].columns:
    print('\n ')
    print('Coluna: ',col, 'Valores: ', df[col].unique())

for col in df[num_columns].columns:
    print('\n ')
    print('Coluna: ',col, 'Valores: ', df[col].min(), df[col].max())

df['TotalCharges'] =  df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] =  df['TotalCharges'].astype(float)

for col in df[num_columns].columns:
    print('\n ')
    print('Coluna: ',col, 'Valores: ', df[col].min(), df[col].max())

print('TotalCharges média:', df['TotalCharges'].mean())
print('tenure média:', df['tenure'].mean())

df['TotalCharges'] =  df['TotalCharges'].replace(0, 2298.06)
df['tenure'] =  df['tenure'].replace(0,29)

df['MonthlyCharges'].hist()
df['tenure'].describe()
df['tenure'].hist()
df.describe()
df['Contract'].describe()
df['Contract'].hist()
df.groupby(["Contract"])["tenure"].mean().plot.bar()
df['Contract'].hist()
df.groupby(['gender'])['gender'].count().plot.pie()
df.groupby(['SeniorCitizen'])['SeniorCitizen'].count().plot.pie()
df.groupby(['PhoneService'])['PhoneService'].count().plot.bar()
df.groupby(['Partner'])['Partner'].count().plot.pie()
df.groupby(['Dependents'])['Dependents'].count().plot.pie()
df['MonthlyCharges'].mean()
df[df['PhoneService']=='Yes']['PhoneService'].count()
df[df['InternetService']!='No']['InternetService'].count()
df[(df['PhoneService']=='Yes') & (df['InternetService']!='No')]['PhoneService'].count()
df.groupby(['InternetService'])['InternetService'].count().plot.bar()

df_contracts = df.groupby(['Contract']).Contract.count()
df_contracts.plot.pie(y='Contract', figsize=(5, 5))

df_payments = df.groupby(['PaymentMethod']).PaymentMethod.count()
df_payments.plot.pie(y='PaymentMethod', figsize=(5, 5))

df.groupby(["gender", "SeniorCitizen", "Partner", "Dependents", "Churn"])["Churn"].count().unstack(['gender', "SeniorCitizen", "Partner", "Dependents"])

df.groupby(["Contract", "Churn"])["Churn"].count().unstack('Contract')
df.groupby(["Contract", "Churn"])["Churn"].count().unstack('Churn').plot.bar()

df.groupby(["PaymentMethod", "Churn"])["Churn"].count().unstack('Churn')

df.groupby(["PaymentMethod", "Churn"])["Churn"].count().unstack('Churn').plot.bar()

df.groupby(['PhoneService'])['PhoneService'].count()

df.groupby('Churn')['customerID'].count()

churns = df['Churn'] == 'Yes'

def churn_0_or_1(value):
    if (value == 'Yes'):
        return 1
    else:
        return 0


df['churn_i'] = df['Churn'].apply(churn_0_or_1)

df.head()

df['churn_i'].mean()

df[df['Churn']=='Yes'][['tenure','MonthlyCharges']].describe().T

df[df['Churn']=='No'][['tenure','MonthlyCharges']].describe().T


df_contracts = df.groupby(['PhoneService']).Contract.count()
df_contracts.plot.pie(y='PhoneService', figsize=(5, 5))

df_contracts = df.groupby(['InternetService']).Contract.count()
df_contracts.plot.pie(y='InternetService', figsize=(5, 5))

df.groupby(["PhoneService", "Churn"])["Churn"].count().unstack('Churn').plot.bar()

df.groupby(["InternetService", "Churn"])["Churn"].count().unstack('Churn').plot.bar()

df[['PhoneService', 'InternetService']].head()

def services(value):
    #print(value)
    if ((value['PhoneService'] == 'Yes') and (value['InternetService'] != 'No')):
        return 'Dois'
    elif (value['PhoneService'] == 'Yes'):
        return 'Somente telefone'
    else:
        return 'Somente internet'
df['services'] = df[['PhoneService', 'InternetService']].apply(services, axis=1)
df.head()

df_contracts = df.groupby(['services']).Contract.count()
df_contracts.plot.pie(y='services', figsize=(5, 5))

df.groupby(["services", "Churn"])["Churn"].count().unstack('Churn').plot.bar()
