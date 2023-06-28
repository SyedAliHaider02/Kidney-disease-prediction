import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('kidney_disease.csv')

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['rbc']=lb.fit_transform(df['rbc'])
df['pc']=lb.fit_transform(df['pc'])
df['pcc']=lb.fit_transform(df['pcc'])
df['ba']=lb.fit_transform(df['ba'])
df['htn']=lb.fit_transform(df['htn'])
df['dm']=lb.fit_transform(df['dm'])
df['cad']=lb.fit_transform(df['cad'])
df['appet']=lb.fit_transform(df['appet'])
df['pe']=lb.fit_transform(df['pe'])
df['ane']=lb.fit_transform(df['ane'])
df['classification']=lb.fit_transform(df['classification'])


df.replace('\t?', float('nan'), inplace=True)  # Replace '\t?' with NaN

# Convert the relevant columns to float
columns_to_convert = [ 'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification']  # Replace with the actual column names
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)



from fancyimpute import KNN

knn_imputer = KNN()
df = knn_imputer.fit_transform(df)

df=pd.DataFrame(df,columns=['id',   'age',   'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification'])


X=df.drop(df['classification'])
y=df['classification'][0:203]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


pickle.dump(rf,open('kidney1.pkl','wb'))