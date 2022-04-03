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


