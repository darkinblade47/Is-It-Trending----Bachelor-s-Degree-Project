"""Modul pentru preprocesarea datelor."""
from pymongo import MongoClient

import re
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """Clasa de preprocesare a datelor."""
    
    def __init__(self, mongo_client, database):
        """
        Constructorul clasei de preprocesare.

        :param mongo_client: Adresa de conexiune cu baza de date MongoDB
        :type mongo_client: str
        :param database: Numele colectiei
        :type database: _type_
        """
        client = MongoClient(mongo_client)
        self.db = client[database]

        self.laptop_full_list = []
        self.scaler_list = [MinMaxScaler((0,1)), MinMaxScaler((0,1)), MinMaxScaler((0,1)), MinMaxScaler((0,1)), MinMaxScaler((0,1))]
        self.pca = PCA(n_components=2)
        self.pca_scaler = MinMaxScaler((0,1))
        self.get_data()

    def get_data(self):
        """Obtinerea datelor din baza de date."""
        query_result = self.db.emag_scrapy.find({})
        for laptop_document in query_result:
            price = laptop_document.get("price")
            scores = laptop_document.get("scores")
            url_link = laptop_document.get("product_url")
            relevance_score = laptop_document.get("relevance_score")
            stock = laptop_document.get("stock")
            # brand  = laptop_document.get("brand")
            specs = laptop_document.get("specification_data")
            memory = -1
            if "Memorie" in specs:
                if "Capacitate memorie" in specs["Memorie"]:
                    memory_string = specs["Memorie"]["Capacitate memorie"].replace(" ","")
                    if memory_string.endswith("GB"):
                        memory = int(re.search(r'\d+', memory_string).group())

            if memory == -1 or len(str(memory))>2:
                continue

            for k in scores:
                scores[k] = float(scores[k]) 

            self.laptop_full_list.append({"price": float(price), 
                                          "relevance_score": float(relevance_score), 
                                          "gpu_score": scores["gpu_score"],
                                          "cpu_score": scores["cpu_score"],
                                          "url":url_link, 
                                          "memory":memory, 
                                          "stock":stock})

    def fit_and_transform(self, feature, min_limit=None, max_limit=None, valid_relevance_score = False):
        """
        Potrivirea datelor pe scaler si apoi normalizarea acestora.

        :param feature: Feature-ul de normalizat
        :type feature: str
        :param min_limit: limita inferioara de pret, defaults to None
        :type min_limit: float, optional
        :param max_limit: limita superioara de pret, defaults to None
        :type max_limit: float, optional
        :param valid_relevance_score: Procesarea datelor cu scor de relevanta peste 5, defaults to False
        :type valid_relevance_score: bool, optional
        :return: Datele normalizare
        :rtype: list(long)
        """
        # relevance_threshold =  5 if valid_relevance_score else 1
        relevance_threshold =  0 if valid_relevance_score else -1
        feature_index = 0 if feature=='gpu_score' else \
                        1 if feature=='cpu_score' else \
                        2 if feature=='relevance_score' else \
                        3 if feature=="price" else \
                        4 if feature=="memory" else 5

        if min_limit is not None and max_limit is not None:
            feature_value_list = np.array([d[feature] for d in self.laptop_full_list if min_limit < d['price'] < max_limit and d['relevance_score'] != relevance_threshold and d["stock"] != -1]) 
        else:
            feature_value_list = np.array([d[feature] for d in self.laptop_full_list if d['relevance_score'] != relevance_threshold and d["stock"] != -1]) 

        feature_value_list[feature_value_list == -1] = 0

        # potrivim modelul partial cu fiecare valoare
        for value in feature_value_list:
            self.scaler_list[feature_index].partial_fit([[value]])
        
        scaled_feature_list = []

        #normalizam valorile dupa potrivire
        for value in feature_value_list:
            scaled_feature_list.append(self.scaler_list[feature_index].transform([[value]])[0][0])
        
        if feature_index == 3:
            return scaled_feature_list, feature_value_list

        return scaled_feature_list

    def process_input_data(self, min_limit=None, max_limit=None, valid_relevance_score = False):
        """
        Procesarea colectiei de date si a altor date de interes necesare.

        :param min_limit: limita inferioara de pret, defaults to None
        :type min_limit: float, optional
        :param max_limit: limita superioara de pret, defaults to None
        :type max_limit: float, optional
        :param valid_relevance_score: Procesarea datelor cu scor de relevanta peste 5, defaults to False
        :type valid_relevance_score: bool, optional
        :return: Datele procesate, link-urile corespunzatoare produselor, preturile nescalate
        :rtype: list(list(float)), list(str), list(float)
        """
        relevance_threshold =  0 if valid_relevance_score else -1
        # relevance_threshold =  5 if valid_relevance_score else 1


        gpu_scores          = self.fit_and_transform("gpu_score", min_limit, max_limit, valid_relevance_score)
        cpu_scores          = self.fit_and_transform("cpu_score", min_limit, max_limit, valid_relevance_score)
        relevance_scores    = self.fit_and_transform("relevance_score", min_limit, max_limit, valid_relevance_score)
        memory_scaled       = self.fit_and_transform("memory", min_limit, max_limit, valid_relevance_score)
        prices, unscaled_prices = self.fit_and_transform("price", min_limit, max_limit, valid_relevance_score)

        input_data          = np.column_stack((gpu_scores, cpu_scores, relevance_scores, prices, memory_scaled))
        pca_scores          = self.pca.fit_transform(input_data)
        self.pca_scores     = self.pca_scaler.fit_transform(pca_scores)

        product_links   = np.array([d["url"] for d in self.laptop_full_list if d['relevance_score'] != relevance_threshold and d["stock"] != -1])

        return input_data, product_links, unscaled_prices

    def unscale_price_center(self, centers):
        """
        Scalarea la valoarea originala a coordonatei de pret a centroizilor pentru plot.

        :param centers: Lista de centroizi
        :type centers: list(list(float))
        :return: Lista de centroizi cu pretul nenormalizat
        :rtype: list(list(float))
        """
        center_copy = np.copy(centers)
        unscaled_price_centers = np.array([self.scaler_list[3].inverse_transform([[d]])[0][0] for d in centers[:,3]])
        center_copy[:, 3] = unscaled_price_centers

        return center_copy

    def unscale_feature(self, feature, value_to_unscale):
        """
        Scalarea la valoarea originala a unui anumit feature.

        :param feature: Feature-ul de adus la valoarea originala
        :type feature: str
        :param value_to_unscale: Valoarea acelui feature
        :type value_to_unscale: float
        :return: Valoarea originala a feature-ului
        :rtype: float
        """
        feature_index = 0 if feature=='gpu_score' else \
                        1 if feature=='cpu_score' else \
                        2 if feature=='relevance_score' else \
                        3 if feature=="price" else \
                        4 if feature=="memory" else 5
        return self.scaler_list[feature_index].inverse_transform([[value_to_unscale]])

    def scale_feature(self, feature, value_to_scale):
        """
        Scalarea unui feature dat.

        :param feature: Feature-ul de scalat
        :type feature: str
        :param value_to_unscale: Valoarea acelui feature
        :type value_to_unscale: float
        :return: Valoarea scalata a feature-ului
        :rtype: float
        """
        feature_index = 0 if feature=='gpu_score' else \
                        1 if feature=='cpu_score' else \
                        2 if feature=='relevance_score' else \
                        3 if feature=="price" else \
                        4 if feature=="memory" else 5
        return self.scaler_list[feature_index].transform([[value_to_scale]])
