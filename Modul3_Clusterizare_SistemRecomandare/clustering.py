import math
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, MeanShift, BisectingKMeans
from sklearn.metrics import silhouette_score

from sklearn.cluster import estimate_bandwidth


class Cluster:  
    """Clasa responsabila de clusterizare si recomandare."""

    def __init__(self, cluster_type, input_data):
        """Constructorul clasei de clusterizare.

        :param cluster_type: Alg de clusterizare
        :type cluster_type: str
        :param input_data: Colectia de date
        """
        self.cluster_type = cluster_type
        self.input_data = input_data
        self.excluded_nodes = []
        self.cluster = None

    def get_cluster(self):
        """
        Efectuarea clusterizarii, setarea atributelor si intoarcerea acestora ca rezultat.

        :return: Obiectul de tip sklearn.cluster, label-urile nodurilor din cluster si in functie de caz, centroizii
        """
        if self.cluster_type == 'dbscan':
            self.cluster = DBSCAN(eps=0.12, min_samples=3)
            self.cluster.fit_predict(self.input_data)
            self.cluster_labels = self.cluster.labels_ 
            return self.cluster, self.cluster_labels, None   # false center
        elif self.cluster_type == 'kmeans':
            self.cluster = KMeans(n_clusters=176, init='k-means++', max_iter=2000)
            self.cluster.fit(self.input_data)
            self.cluster_labels = self.cluster.labels_
            cluster_centers = self.cluster.cluster_centers_
            return self.cluster, self.cluster_labels, cluster_centers
        elif self.cluster_type=='birch':
            self.cluster = Birch(n_clusters=3)
            self.cluster.fit(self.input_data)
            self.cluster_labels = self.cluster.labels_
            return self.cluster, self.cluster_labels, None # false center
        elif self.cluster_type=='aglomerative':
            self.cluster = AgglomerativeClustering(
                    n_clusters=None, linkage="ward", distance_threshold=0.36                )
            self.cluster.fit(self.input_data)
            # plt.title("Hierarchical Clustering Dendrogram")
            # plot_dendrogram(aglomerative, truncate_mode="level", p=3)
            # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            # plt.show()

            self.cluster_labels = self.cluster.labels_

            return self.cluster, self.cluster_labels, None # false center
        elif self.cluster_type=='meanshift':
            band = estimate_bandwidth(self.input_data, quantile=0.5)
            self.cluster = MeanShift(bandwidth=band, bin_seeding=True)
            self.cluster.fit(self.input_data)
            self.cluster_labels = self.cluster.labels_
            cluster_centers = self.cluster.cluster_centers_
            return self.cluster, self.cluster_labels, cluster_centers
        elif self.cluster_type=='bisect':
            # band = estimate_bandwidth(self.input_data, quantile=0.5)
            self.cluster = BisectingKMeans(n_clusters=8, init="k-means++")
            self.cluster.fit(self.input_data)
            self.cluster_labels = self.cluster.labels_
            cluster_centers = self.cluster.cluster_centers_
            return self.cluster, self.cluster_labels, cluster_centers
        
    def compute_cluster_centroids(self, cluster_data):
        """
        Calcularea centroizilor prin medie.

        :return: Lista cu coordonatele centroizilor
        :rtype: list(list(float))
        """
        labels = self.cluster.labels_
        self.unique_labels, label_counts = np.unique(labels, return_counts=True)

        self.centroids = np.zeros((len(self.unique_labels), cluster_data.shape[1]))
        self.non_pca_centroids = np.zeros((len(self.unique_labels), self.input_data.shape[1]))
        for i, label in enumerate(self.unique_labels):
            cluster_points = cluster_data[labels == label]
            non_pca_cluster_points = self.input_data[labels == label]
            self.centroids[i] = np.mean(cluster_points, axis=0)
            self.non_pca_centroids[i] = np.mean(non_pca_cluster_points, axis=0)
        
        # clusterele din care nu putem da recomandari deoarece au relevanta < 1 (normalizată ca 0.2)
        self.excluded_nodes = [index for index, d in enumerate(self.non_pca_centroids) if d[2] < 0.2]
        return self.centroids

    def compute_closest_better_centroid(self, cluster_id):
        """
        Cautarea celui mai apropiat centroid de centroidul cu ID=cluster_id.

        :param cluster_id: Clusterul pentru care cautam cel mai apropiat centroid
        :type cluster_id: int
        :return: Lista cu cei mai apropiati centroizi valizi
        :rtype: list(int)
        """
        eps = 0.01

        closest_ids = []
        closer_centroid_id = None
        closer_distance = float("inf")

        current_centroid = self.centroids[cluster_id]

        for index, c in enumerate(self.centroids):

            if (index == cluster_id) or (index in self.excluded_nodes):
                continue

            #cautam un produs cu o performanta mai buna sau relativ identice
            if (c[0] >= current_centroid[0] or c[1] >= current_centroid[1] or c[4] >= current_centroid[4]) and c[2] >= current_centroid[2]:
                distance_between_clusters = math.dist(c, current_centroid) #distanta euclidiana intre centroid-ul nostru si cel curent din for
                if closer_distance == float('inf'): 
                    closer_distance = distance_between_clusters
                    closer_centroid_id = index
                    closest_ids.append({index:distance_between_clusters})
                    continue
                
                if closer_distance - eps <= distance_between_clusters <= closer_distance + eps:
                    if distance_between_clusters < closer_distance:
                        closer_distance = distance_between_clusters
                    closest_ids.append({index:distance_between_clusters})
                elif distance_between_clusters < closer_distance:
                    closer_distance = distance_between_clusters
                    closest_ids = []
                    closest_ids.append({index:distance_between_clusters})
                    closer_centroid_id = index

        if len(closest_ids) != 0:
            # verific ca potentialele clustere de recomandare sa nu se fi indeparatat prea mult de clusterul curent
            closest_ids = [next(iter(x.keys())) for x in closest_ids if next(iter(x.values())) <= closer_distance + eps]
            if len(closest_ids) > 1:
                print(f"Cele mai apropiate clustere de {cluster_id} sunt:")
                print(closest_ids)
                return closest_ids
        
        print(f"Cel mai apropiat cluster de {cluster_id} este: {closer_centroid_id}")
        return [closer_centroid_id]
