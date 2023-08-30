import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from clustering import Cluster
from preprocessing import DataProcessor
#====================================== CONSTANTS ==================================
mongo_client_address = "mongodb://admin:passwdadmin@192.168.56.20:27017"
database_name= "EmagProductDatabase"
PLOTTING = True
#===================================================================================

class Recommendation:
    """Clasa de recomandare."""

    def __init__(self):
        """Constructorul clasei de recomandare."""
        self.preprocessing_obj = DataProcessor(mongo_client_address, database_name)
        self.input_data, self.product_links, unscaled_prices = self.preprocessing_obj.process_input_data(valid_relevance_score=False)

        self.cluster_obj = Cluster("aglomerative", self.input_data)
        self.cluster, self.cluster_labels, cluster_centers = self.cluster_obj.get_cluster()
        self.cluster_centers = self.cluster_obj.compute_cluster_centroids(self.input_data)
        self.cluster_centers = self.preprocessing_obj.unscale_price_center(self.cluster_centers)
        self.recommendation_graph = self.create_recommendation_graph()
    
    def get_recommendation_links(self, product_link, req_min_range, req_max_range):
        """
        Obtinerea link-urilor de recomandare pentru produsul de interes in range-ul de pret specificat.

        :param product_link: Link-ul url al produsului.
        :type product_link: str
        :param req_min_range: Prag minim de pret pentru recomandari.
        :type req_min_range: float
        :param req_max_range: Prag maxim de pret pentru recomandari.
        :type req_max_range: flaot
        :return: Link-urile recomandarilor.
        :rtype: list(str)
        """
        laptop_index = np.where(self.product_links == product_link)

        cluster_id = self.cluster_labels[laptop_index]

        min_range = self.preprocessing_obj.scale_feature("price", req_min_range)[0][0]
        max_range = self.preprocessing_obj.scale_feature("price", req_max_range)[0][0]

        results = self.get_recommendations(cluster_id, (min_range, max_range))
        recommendation_links = {"similar":set(), "recommended":set()}

        for type_of_laptop in results:
            for index_of_laptop in results[type_of_laptop]:
                recommendation_links[type_of_laptop].add(self.product_links[index_of_laptop])

        return recommendation_links


    def create_recommendation_graph(self):
        """
        Crearea grafului de recomandare.

        :return: graful de recomandare
        :rtype: nx.DigGraph
        """
        graph = nx.DiGraph()

        for i in self.cluster_obj.unique_labels:
            graph.add_node(i)

        for i in self.cluster_obj.unique_labels:
            rec = self.cluster_obj.compute_closest_better_centroid(i)
            for j in rec:
                if j is not None:
                    graph.add_edge(i, j)
    
        # pos = nx.spring_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue',  edge_color='gray', font_size=24)
        # edge_labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        # plt.show()

        return graph

    def get_recommendations(self, cluster_id, price_range):
        """
        Se obtine graful de recomandare, se parcurge folosind BFS, astfel se obtin subliste cu indexii recomandarilor, urmand a fi extrasi cei mai buni din fiecare sublista folosind round robin.

        :param cluster_id: clusterul pentru care cautam recomandari
        :type cluster_id: int
        :param price_range: intervalul de pret in care cautam recomandari
        :type price_range: tuple
        :return: dictionar cu liste de indexi: indexi pentru laptopuri similare, liste pentru laptopuri recomandate
        :rtype: dict{str: set(int)}
        """
        recommendation_graph = self.recommendation_graph
        recommendations_results = []
        # Parcurgere BFS
        visited = set()  
        queue = [cluster_id] 
        current_count = 0

        while queue:
            current_node = queue.pop(0)
            if isinstance(current_node, np.ndarray):
                current_node = current_node[0] # [0] fiindca primesc tuple (array, type)

            if current_node in visited:
                continue

            visited.add(current_node)

            print("Căutam recomandări în nodul: ", current_node)
            is_starting_node = current_node == cluster_id
            recommendations_results.append(self.obtain_recommendations(current_node, price_range, is_starting_node))
            
            neighbors = recommendation_graph.neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        for d in recommendations_results:
            for value_list in d.values():
                current_count += len(value_list)
            
        result = {"similar":set(), "recommended": set()}
        counter = 0
        sublist_lengths = [sum(len(vals) for vals in d.values()) for d in recommendations_results]
        max_length = max(sublist_lengths)

        #in mod normal recomand maxim 15 laptop-uri, dar daca sunt mai multe noduri de recomandare,
            #maresc numarul de recomandari astfel incat sa am cate un laptop din fiecare nod 
        max_recommendations = len(sublist_lengths) if len(sublist_lengths) > 15 else 15

        #round robin
        for i in range(max_length):
            for index, sublist in enumerate(recommendations_results):
                key = "similar" if index==0 else "recommended" #daca e sublista de laptop-uri similare
                if i < len(sublist[key]):
                    result[key].add(sublist[key][i])
                    counter += 1
                    if counter == max_recommendations:
                        break
            if counter == max_recommendations:
                break
        
        return result

    def obtain_recommendations(self, cluster_id, price_range, is_starting_node):
        """
        Se obtin indexii laptopurilor de tip recomandare intr-un anumit interval de pret, ordonati dupa relevanta lor.

        :param cluster_id: clusterul pentru care cautam recomandari
        :type cluster_id: int
        :param price_range: intervalul de pret in care cautam recomandari
        :type price_range: tuple
        :return: lista de subliste de indexi
        :rtype: list(list(int))
        """

        min_price = price_range[0]
        max_price = price_range[1]

        #obtin indexii laptopurilor care apartin clusterului clusterID
        indexes_of_laptops_from_cluster_id = [index for index,d in enumerate(self.cluster_labels) if d == cluster_id]
        
        laptop_cluster = []
        for index in indexes_of_laptops_from_cluster_id:
            current_laptop = self.input_data[index]
            if min_price <= current_laptop[3] <= max_price:
                laptop_cluster.append((index, self.input_data[index]))

        sorted_by_relevance = sorted(laptop_cluster, key=lambda x: x[1][2]) #laptop_cluster are forma (index, [cpu_performace, gpu_performance, relevanta, pret, memory])
        key = "similar" if is_starting_node else "recommended"
        indexes = [d[0] for d in sorted_by_relevance] #indexii globali
        dictionary = {key:indexes} #dictionar de separare a laptopurilor similare/recomandate
        return dictionary 