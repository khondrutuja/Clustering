# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:13:27 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np

univ1 = pd.read_excel("C:/2-dataset/University_Clustering(1).xlsx")
univ1
a = univ1.describe()
##################EDA#########################
univ1.columns
#Now we have one column that is state which not really very usefull
#So just drop it
# Q. How to identify the which column is usless or not requred
#---> 

univ = univ1.drop(['State'],axis = 1)
univ
#We know that there is scale difference among the columns ,which we have
#to remove either by normalization or by standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function to univ dataframe
#for all the rows and columns from 1 until end 
#since 0th column has university name hence skipped 
df_norm = norm_func(univ.iloc[:,1:])
df_norm
#You can check the df_norm dataframe which id scalled
#Between values from 0 to 1
#you can apply describe() frunction 

b = df_norm.describe()
#before you apply clustering you need to plot dendrogtam first 
#now to create dendrogram ,we need to measure distance 
#We have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#scipy.cluster.hierarchy is a sub-module within the SciPy library
#that provides functionality for hierarchical clustering and 
#dendrogram plotting.

#linkage is a function provided by this 
#module that is used to perform hierarchical clustering.

#linkage function gives us hierarchical or aglomerative clustering
#Ref the help for linkage 
import matplotlib.pyplot as plt
z = linkage(df_norm,method='complete',metric = 'euclidean')
#z = linkage(df_norm, method='complete', metric='euclidean'):
#linkage function computes the linkage matrix 
#for hierarchical clustering. It takes several arguments:
plt.figure(figsize =(15,8))
plt.title("Hierarchical clustering dendrogram");
plt.xlabel("Index")
plt.ylabel("distance")
#ref help of dendrogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation = 0,leaf_font_size = 10)
#z=linkage matrix,
plt.show()
#dendrogram()
#applying agglomerative clustering choosing 3 as clusters
#from dendrogram
#whatever has been displayed in dendrogram is not clustering 
#it is just showing number of possible cluster

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels tothe clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to univ dataframe as column and name the column
univ['clust'] = cluster_labels
#we want to relocate the column 7 to 0th position 
univ1 = univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the univ1 dataframe 
univ1.iloc[:,2:].groupby(univ1.clust).mean()
#from the output cluster 2 has got highest top 10 
#lowest accept ratio ,best faculty ratio and highest expenses
#highest graduates ratio
univ.to_csv("University.csv",encoding = 'utf-8')
import os
os.getcwd()#checks the current working directory using os.getcwd()







