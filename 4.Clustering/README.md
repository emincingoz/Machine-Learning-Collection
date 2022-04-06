[Scikit-Learn Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html)

![image](https://user-images.githubusercontent.com/49842813/161968938-dcb6f7b0-afad-4294-89aa-a59b300aeea5.png)


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
