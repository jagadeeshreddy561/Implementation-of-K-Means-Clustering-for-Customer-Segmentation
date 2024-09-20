## Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## STEP 1 :
Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

## STEP 2 :
Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

## STEP 3 :
Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

## STEP 4 :
Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

## STEP 5 :
Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

## STEP 6 :
Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

## STEP 7 :
Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
Program to implement the K Means Clustering for Customer Segmentation.
## Developed by: Mallu Jagadeeswar Reddy
## RegisterNumber:212222240059  
```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## output:

## data.head() :



![ml-8 1](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/73ba80fc-def3-4fe7-9546-34b300548ad6)


## data.info() :


![ml-8 2](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/9a43b4db-c57f-4783-8de0-65383e91a14a)



## Null Values :


![ml-8 3](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/d073f745-dbfa-4550-bd70-6b8d78548a82)



## Elbow Graph :


![ml-8 4](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/c2e979a2-5703-4ce4-9f20-085907f06216)


## K-Means Cluster Formation :


![ml-8 5](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/f44947ca-9531-48bb-8bb3-fe6bf3cd4087)


## Predicted Value :


![ml-8 6](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/c9381a26-3358-4cc2-93d9-dfd12e1a9894)



## Final Graph :

![ml-8 7](https://github.com/jagadeeshreddy561/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/120623104/2a80c091-9132-46a6-8960-eca6427db035)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
