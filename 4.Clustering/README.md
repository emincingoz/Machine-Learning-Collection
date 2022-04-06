[Scikit-Learn Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html)

![image](https://user-images.githubusercontent.com/49842813/161968938-dcb6f7b0-afad-4294-89aa-a59b300aeea5.png)


# K-Means (Unsupervised Learning)

* How many clusters there will be is taken from the user. (Disadvantage)
* k center points are choosen *randomly*
* Each data sample is assigned to the corresponding cluster according to the nearest centroid.
* For each cluster formed, the center points is calculated again.


![image](https://user-images.githubusercontent.com/49842813/161224744-b5b27536-6403-4f97-bf5b-f7070d466674.png)


![image](https://user-images.githubusercontent.com/49842813/161224518-584e1894-9810-4c20-8186-f9142186103f.png)


![image](https://user-images.githubusercontent.com/49842813/161225110-fbe08c09-457c-4e04-926f-05dfea797210.png)


## Random Initialization Trap in K-Means

Random initialization trap is a problem that occurs in the K-Means algorithms. 

In random initialization trap when the centroids of the clusters to be generated randomly then inconsistency may be created and this may sometimes lead to generating wrong clusters in the dataset. So random initialization may sometimes prevent us from developing the correct clusters.

![image](https://user-images.githubusercontent.com/49842813/161447065-73664df5-93cb-499a-8479-ddb332839d6c.png)  ![image](https://user-images.githubusercontent.com/49842813/161447076-1cd117ac-8b18-4d15-b04f-09643a30b962.png)

The distribution of the centroids above will give results that may be erroneous compared to the distribution below.

![image](https://user-images.githubusercontent.com/49842813/161447121-2dda3493-ddd8-4507-8302-a33b5ad9c86c.png)  ![image](https://user-images.githubusercontent.com/49842813/161447130-1650f94f-a5b7-4116-a87c-6df6a0202bce.png)

# K-Means++

To overcome the *random initialization trap* we use K-Means++. This algorithm ensures a smarter initialization of the centroids and improves the quality of the clusters. Apart from the initialization, the rest of the algorithm is the same as the standart K-Means algorithm. Steps:

1. Randomly select the first centroid from the data points.
2. For each data point computs its distance from the nearest, previously choosen centroid.
3. Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously choosen centroid (The point having maximum distance from the nearest centroid is most likely to be selected next as a centroid.)
4. Repeat steps 2 and 3 until k centroids have been sampled.

## WCSS
WCSS (Within-Cluster Sums of Squares)

![image](https://user-images.githubusercontent.com/49842813/161447828-fd59ffb9-2931-440f-a66f-51b902acf0f3.png)


---

# Agglomerative Clustering

* Initialy, all data is assumed as a single set.
* A cluster created by taking the two closest data points.
* A new cluster is created by taking the two closest clusters.

## Distance Measurement (Between Clusters)

According to the selected distance method (euclidean, manhattan, chebyshev, mahalanobis, minkowski distance, ...):

* The distance between the closest points can be taken between the clusters
* The distance between the farthest points can be taken between the clusters
* The distance of all points in the clusters are taken and added together and the average of this sum can be taken.
* The distance between the midpoints of the clusters can be taken.
* In this way the previous step continues until there is only one cluster. (The process can be terminated when the specified number of k-value clusters are created.)


![image](https://user-images.githubusercontent.com/49842813/161972540-875b9187-c137-4bdd-93bb-5f0eab3dc5bd.png)
