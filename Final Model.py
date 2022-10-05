import pandas as pd
import numpy as np
import warnings
import pickle
from collections import Counter
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

df = pd.read_csv("G:\My Drive\College Project\Bank Churners\BankChurners.csv")

df = df.drop(["CLIENTNUM"],axis = 1)

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df = remove_outlier(df,"Customer_Age")
df = remove_outlier(df,"Months_on_book")
df = remove_outlier(df,"Months_Inactive_12_mon")
df = remove_outlier(df,"Contacts_Count_12_mon")
df = remove_outlier(df,"Credit_Limit")
df = remove_outlier(df,"Avg_Open_To_Buy")
df = remove_outlier(df,"Total_Amt_Chng_Q4_Q1")
df = remove_outlier(df,"Total_Trans_Amt")
df = remove_outlier(df,"Total_Trans_Ct")

df.Attrition_Flag = df.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
df.Gender = df.Gender.replace({'F':1,'M':0})
df = pd.concat([df,pd.get_dummies(df['Education_Level']).drop(columns=['Unknown'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['Income_Category']).drop(columns=['Unknown'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['Marital_Status']).drop(columns=['Unknown'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['Card_Category']).drop(columns=['Platinum'])],axis=1)
df.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category'],inplace=True)

req_cols = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio']
            
scaler = StandardScaler()
df[req_cols] = scaler.fit_transform(df[req_cols])

y = df.pop("Attrition_Flag")
x = df


sm = SMOTE(random_state = 69, sampling_strategy = 1.0)
x, y = sm.fit_resample(x, y)


Top10Features = ['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Months_Inactive_12_mon','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count',
                 'Avg_Utilization_Ratio',
                 'Total_Amt_Chng_Q4_Q1',
                 'Credit_Limit',
                 'Contacts_Count_12_mon']
x = x[Top10Features].values
y = y.values

model = RandomForestClassifier(criterion='entropy', max_depth = 24, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 7, n_estimators = 143)
model.fit(x,y)


pickle.dump(model,open("model.pkl","wb"))