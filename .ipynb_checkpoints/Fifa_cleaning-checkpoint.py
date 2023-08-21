# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:21:52 2019

@author: mac
"""

import os
import pandas as pd
import numpy  as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
os.getcwd()
os.chdir('/Users/mac/Desktop/R/')
fdf=pd.read_csv('football.csv')
fdf.info()
fdf.shape#number of columns and rows before droping
colist=list(fdf.columns.values)
print(*fdf, sep='\n')#getting the coulmn names 
#preprocessing droping incessery
fdf.drop(['ID','Photo','Flag','Body Type','Loaned From','Work Rate','Contract Valid Until','Release Clause','International Reputation','LS','ST','RS','LW','LF','RB','RCB','CB','LB','RWB','RDM','CDM','LDM','LWB','RM','RCM','CM','LCM','LM','RAM','CAM','LAM','RW','RF','CF','LCB','Joined','Unnamed: 0','Special','Release Clause', 'Crossing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','Agility','Jumping','LongShots','Aggression','Interceptions','Club Logo'],inplace=True,axis=1)
fdf.drop(['Club Logo'],inplace=True,axis=1)

fdf.to_csv('fifa_players.csv', index=False)
fdf.columns
fdf.isnull().sum().sum()
#preprocessing: 
fdf.isnull().any()
#Missing data in form of white lines 
#msno.matrix(fdf,color=(0.2,0,0.9))
fdf.info()
fdf.describe().T
#1- renaming columns withoutspace
fdf=fdf.rename(columns={'Skill Moves':'SkillMoves','Jersey Number':'JerseyNumber',})
#####
def extract_value_from(Value):
    out = Value.replace('€', '')
    if 'M' in out:
        out = float(out.replace('M', ''))*1000000
    elif 'K' in Value:
        out = float(out.replace('K', ''))*1000
    return float(out)


fdf['Value'] = fdf['Value'].apply(lambda x: extract_value_from(x))
fdf['Wage'] = fdf['Wage'].apply(lambda x: extract_value_from(x))


#2- : Function for taking out the "lbs" from weight & replace null values with the mean 
def weight_correction(df):
    try:
        value = float(df[:-3])
    except:
        value = 0
    return value
fdf['Weight'] = fdf.Weight.apply(weight_correction)
fdf['Weight'].fillna(fdf.Weight.mean(), inplace = True)
fdf['Weight'].isnull().sum().sum()
#3- checked null values and drop na where the Poistion is equal nan
#fdf.loc['Position' == 'GK'].apply.fillna(value=0,inplace=True)
fdf.isnull().sum()
fdf.isnull().values.any()
fdf.isnull().sum().sum()#total sumation of missing values 
#fuction for catograizing the poistion feature 
def position_simplifier(val):
    if val == 'RF' or val == 'ST' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'CF':
        val = 'Forword'
        return val
    elif val == 'LW' or val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'RM' \
         or val == 'LAM' or val == 'LM' or val == 'RDM' or val == 'RW' or val == 'CM' or val == 'RAM':
        val = 'Midfilder'
        return val    
    elif val == 'RCB' or val == 'CB' or val == 'LCB' or val == 'LB' or val == 'RB' or val == 'RWB' or val == 'LWB':
        val = 'Defense'
        return val
    else:
        return val
fdf['Position']=fdf.Position.apply(position_simplifier)
fdf.dropna(subset=['Position'],inplace=True) 
fdf['Position'].isnull().sum().sum()
fdf['Position'].value_counts().plot(kind='pie',autopct='%1.1f%%',title='Poistion Distribution')
plt.figure(figsize = (20, 10))
sns.countplot(x=fdf.Position, data=fdf)
plt.show()

#4- Check the avarage for hieghts between 5.9 and 6.11 from height over data distribution 
def height_converter(val):
    f = val.split("'")[0]
    i = val.split("'")[1]
    h = (int(f) * 30.48) + (int(i)*2.54)
    return h
fdf['Height'].mode()
fdf['Height'].isnull().sum().sum()
fdf['Height'].fillna("5'11", inplace = True)
fdf['Height'] = fdf['Height'].apply(height_converter)
s=fdf['Height'][0]+fdf['Height'][1]
#5 Display correlation 
corr=fdf.corr()
type(corr)
#make dummies for Position column code 
fdf_dum = pd.get_dummies(fdf,columns=['Position'],drop_first=True)

#Scaling the datasets 

sc = StandardScaler()
fdf_scaled = pd.DataFrame(sc.fit_transform(fdf_dum), columns=fdf_dum.columns)

fdf_dum.isnull()
fdf_dum.isnull().sum().sum()
# Establish features and response 
x = fdf_scaled.drop(['Overall'], axis = 1)
y = fdf_scaled.Overall

# Create train set with 70%, test set with 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
type(x_train)



X1=sm.add_constant(x_train) 
ols=sm.OLS(y_train,X1)
lr=ols.fit()
lr.pvalues
pvalue=max(lr.pvalues[1:len(lr.pvalues)])
print(lr.summary())
while (pvalue>=0.05):
    loc=0
    for i in lr.pvalues:
        if (i==pvalue):
            feature=lr.pvalues.index[loc]
            print(feature)
        loc+=1
    x_train=x_train.drop(feature,axis=1)
    x_test=x_test.drop(feature,axis=1)
    X1=sm.add_constant(X_train) 
    ols=sm.OLS(y_train,X1)
    lr=ols.fit()
    pvalue=max(lr.pvalues[1:len(lr.pvalues)])
    print(lr.summary())






















x1 = sm.add_constant(x_train)
ols = sm.OLS(y_train, x1)
lr = ols.fit()

print(lr.summary())
#As p value are less than 0.05
# Model Traning
model = LinearRegression()
model.fit(x_train, y_train)

#  Model Testing
y_pred = model.predict(x_test)

# y intercept
print(model.intercept_)

# cofficients for all x variables 
#DESCRIPE the mathetcial realtionshipe between independnt variable and dependents
print(model.coef_)
cofficients = model.coef_
#What should be the range of MSE
print(mean_squared_error(y_test, y_pred))

#Can MSE be high with low R2 and vice-versa

print((r2_score(y_test, y_pred)))
print(r2_score(y_train,model.predict(x_train)))


#Check correlation among players and thier skills 
#Lowest correlation among the goalkeeping side with other columns and high among themselves 
#High correlation between Dribbling, Volleys, Passing etc..
plt.figure(figsize=(50,50))
sns.heatmap(fdf.corr(),linewidths=0.1,linecolor='black',square=True,cmap='summer')



#######

'''sns.pairplot(fdf_scaled,x_vars=fdf_scaled.columns,y==fdf_scaled.columns,kind='reg')
'''
###########
x_test.shape
y_test

# Visualising test dataset
plt.figure(figsize=(18,10))
sns.regplot(y_test,y_pred,scatter_kws={'alpha':0.3,'color':'blue'},line_kws={'color':'black','alpha':0.5})
plt.xlabel('Predictions')
plt.ylabel('Overall')
plt.title("Player Rating Predction")
plt.show()

#k-Fold 
model = LinearRegression()
scores = []
kfold = KFold(n_splits=4, shuffle=True, random_state=23)
for i, (train, test) in enumerate(kfold.split(x, y)):
 model.fit(x.iloc[train], y.iloc[train])
 score = model.score(x.iloc[test], y.iloc[test])
 scores.append(score)
print(scores)

'''As we see all the P- values are less than SL(0.05), that means all the variables are significant and none of them can be removed. t-value shows the statistical significane of each variable. F-static shows us how significant the fit is. Adjusted- R is 0.93 that means our model explains 92.0% variables in dependent variables'''
plt.scatter(y_test,y_pred)
plt.xlabel("OverAll test")
plt.ylabel("OverAll Predict")
plt.show()

# print linear regresstion
'''
from sklearn.model_selection import cross_val_score
clf = LinearRegression()
cross_val_score(clf,x,y,cv=4).mean() 
'''











