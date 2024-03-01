# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:18:47 2023

@author: Dell
"""
####Very Very IMP ############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#before dropping any column check the heat map and check the correlation of that column 
#let us try to understand first how k means works for two
#dimentional data
#for that ,generate random numbers in the range 0 to 1
#and with uniform probability of 1/50
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
#create a empty dataFrame with 0 rows and 2 columns 
df_xy = pd.DataFrame(columns = ['X','Y']) 
#Assign the values of x and y to these columns
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x='X',y='Y',kind = "scatter")
model1=KMeans(n_clusters = 3).fit(df_xy)
'''
with data x and y ,apply kmeans model
generate scatter plot with scale/font  = 10

cmap=plt.cm.coolwarm:cool color combination
'''
model1.labels_
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',s=10,cmap=plt.cm.coolwarm)

##########################################################3

univ1 = pd.read_excel("University_Clustering.xlsx")
univ1.describe()
univ = univ1.drop(['State'],axis=1)
#we know that there is scale difference among the the columns,which we have
#either by using normalization or standardization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to univ dataframe for all the rows

df_norm = norm_func(univ.iloc[:,1:])

'''
What will be ideal cluster number,will it be 1,2 or 3 
'''

TWSS=[]
k = list(range(2,8))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_) #Total within sum of sum
'''
    KMeans inertia ,also known as sum of squares errors(or SSE) ,
    colculates the sum of the distances of all points
    
    

'''
TWSS
#as k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total_within_SS")

'''
How to select value of k from elbow cusrve 
when l changes from 2 to 3 ,then decreases in twss is higher han when k changes 
from 3 to 4 .
when k changes from 5 to 6 decreases in TWSSS is considerablly less ,hence consider k=3

'''

model = KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb = pd.Series(model.labels_)
univ['clust'] = mb
univ.head()
univ = univ.iloc[:,[7,0,1,2,3,4,5,6]]
univ
univ.iloc[:,2:8].groupby(univ.clust).mean()
univ.to_csv("kemans_university.csv",encoding = 'utf-8')
import os
os.getcwd()