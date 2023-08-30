"""Fișierul care execută clusterizarea și generează un plot"""

from cluster_plot import Plot
from preprocessing import DataProcessor
from clustering import Cluster
import numpy as np

mongo_client_address = "mongodb://admin:passwdadmin@192.168.56.20:27017"
database_name= "EmagProductDatabase"

TWO_D = False
if  __name__ == '__main__':
    preprocessing_obj = DataProcessor(mongo_client_address, database_name)
    input_data, product_links, unscaled_prices = preprocessing_obj.process_input_data(valid_relevance_score=True)
    pca_for_plot = preprocessing_obj.pca_scores
    entire_data = np.column_stack((input_data, product_links))
    cluster_obj = Cluster("aglomerative", input_data)
    cluster, cluster_labels, cluster_centers =  cluster_obj.get_cluster()
    cluster_centers = cluster_obj.compute_cluster_centroids(pca_for_plot)
    non_pca_centroids = cluster_obj.non_pca_centroids

    non_pca_centroids = preprocessing_obj.unscale_price_center(non_pca_centroids)
    unique_labels, label_counts = np.unique(cluster_obj.cluster_labels, return_counts=True)

    if True:    
        plot_obj = Plot()
        if TWO_D is False:
            if cluster_centers is not None:
                plot_obj.init_3d_plot(input_data=pca_for_plot, product_links=product_links, unscaled_prices=unscaled_prices, cluster_labels=cluster_labels, cluster_centers=(cluster_centers,non_pca_centroids))
            else:
                plot_obj.init_3d_plot(input_data=pca_for_plot, product_links=product_links, unscaled_prices=unscaled_prices, cluster_labels=cluster_labels)
        else:
            if cluster_centers is not None:
                plot_obj.init_2d_plot(input_data=pca_for_plot, product_links=product_links, cluster_labels=cluster_labels, cluster_centers=cluster_centers)
            else:
                plot_obj.init_2d_plot(input_data=pca_for_plot, product_links=product_links, cluster_labels=cluster_labels)
