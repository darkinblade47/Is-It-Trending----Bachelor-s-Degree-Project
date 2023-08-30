# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import datetime

import os
import sys
import math
import pymongo
import datetime
import requests
from fuzzywuzzy import fuzz
from numpy import mean, floor
from dateutil.relativedelta import relativedelta
from scrapy.utils.project import get_project_settings

FOLDER_PATH = os.path.dirname(__file__)

sys.path.append(os.path.join(FOLDER_PATH, "../../../Modul1_SentimentAnalysis"))

from predicter import SentimentPredict

class EmagPipeline(object):
    """
    Pipeline de procesare pentru produse din eMag

    :param object: object
    :type object: object
    """

    def __init__(self, mongo_uri, mongo_db, mongo_collection) -> None:
        self.extracted_review_data = []
        self.extracted_specification_data = []
        self.extracted_gpu_scores = {}
        self.extracted_cpu_scores = {}
        self.predict_class = SentimentPredict()
        self.final_data = []

        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
    

    @classmethod
    def from_crawler(cls, crawler):
        """
        Preluarea datelor de autentificare la baza de date la pornirea scraper-ului.

        :param crawler: crawler
        :type crawler: crawler
        :return: 
        :rtype: 
        """
        return cls(
            mongo_uri=crawler.settings.get("MONGODB_URI"),
            mongo_db=crawler.settings.get("MONGODB_DATABASE"),
            mongo_collection=crawler.settings.get("MONGODB_COLLECTION")
        )


    def open_spider(self, spider):
        """
        Funcție apelată la pornirea unui spider.

        :param spider: spider de extragere a datelor
        :type spider: spider
        """
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.database = self.client[str(self.mongo_db)]
        self.collection = self.database[str(self.mongo_collection)]

    def close_spider(self, spider):
        """
        Funcție apelată la încheiera extragerii datelor unui spider.

        :param spider: spider de extragere
        :type spider: spider
        """
        self.process_data()
        self.process_relevance_rating()

        self.manage_mongo_data()
        self.send_recluster_trigger()
        
        self.client.close()

    def send_recluster_trigger(self):
        """Funcție apelată pentru a trimite un trigger serverului Flask."""
        remaining_tries = 5
        while remaining_tries >= 0:
            response = requests.post("http://localhost:5000/recluster")
            if response.status_code == 200:
                print('Clusterizare efectuată cu succes!')
                remaining_tries = -1
            else:
                print('Reclusterizarea a eșuat. Status code:', response.status_code)
                remaining_tries -= 1


    def process_item(self, item, spider):   
        """
        Funcție apelată la fiecare yield efectuat de spider.

        :param item: rezultatul primit de la spider
        :type item: object
        :param spider: spider-ul ce returnează datele
        :type spider: spider
        :return: item
        :rtype: object
        """
        if 'extracted_reviews' in item:
            self.extracted_review_data.append(item['extracted_reviews']['reviews']) 

        if 'specifications' in item:
            self.extracted_specification_data.append(item['specifications'])

        if 'cpu_score' in item:
            self.extracted_cpu_scores = item['cpu_score']

        if 'gpu_score' in item:
            self.extracted_gpu_scores = item['gpu_score']

        return item  


    def process_data(self):
        """Funcția care începe procesul de prelucrare a datelor."""
        merged_review_data = self.merge_review_data()

        for product_id in merged_review_data:
            sentiments = self.get_sentiment_of_reviews(merged_review_data[product_id])
            for i in range(len(merged_review_data[product_id])):
                merged_review_data[product_id][i]['review_content'] = sentiments[i]
        
        merged_sentiment_data = {k: [{k2 if k2 != 'review_content' else 'sentiment': v2 for k2, v2 in d.items()} for d in v] for k, v in merged_review_data.items()}
        self.merge_data(merged_sentiment_data)


    def get_sentiment_of_reviews(self, review_data):
        """
        Funcția care apelează modelul de predicție al sentimentelor.

        :param review_data: recenziile
        :type review_data: list
        :return: sentimentul recenziilor
        :rtype: list
        """
        content_for_predict = [f'{d["title"]} {d["review_content"]}' for d in review_data]
        return self.predict_class.predict(content_for_predict)


    def merge_review_data(self):
        """
        Funcția de concatenare a recenziilor per produs.

        :return: dicționare de recenzii per produs
        :rtype: dict
        """
        concatenated_dict = {}
        for intermediary_review_list in self.extracted_review_data:
            for product_id in intermediary_review_list:
                if f"{product_id}" in concatenated_dict:
                    concatenated_dict[f"{product_id}"].extend(intermediary_review_list[product_id])    
                else:
                    concatenated_dict[f"{product_id}"] = intermediary_review_list[product_id]

        merged_data_by_product_id = concatenated_dict
        return merged_data_by_product_id
    

    def merge_data(self, processed_reviews):
        """
        Concatenarea recenziilor cu specificațiile și scorurile de performanță.

        :param processed_reviews: recenziile cu sentiment
        :type processed_reviews: list
        """
        keys_indexes_of_specifications = {}
        for d in self.extracted_specification_data:
            keys_indexes_of_specifications[d['product_id']] = d
        
        for key in keys_indexes_of_specifications:
            new_product = keys_indexes_of_specifications[key].copy()
            if key in processed_reviews:
                new_product["review_data"] = processed_reviews[key]
            else:
                new_product["review_data"] = []
            cpu_score, gpu_score = self.process_specification_full_names(new_product)
            new_product['scores'] = {'cpu_score':cpu_score, 'gpu_score':gpu_score}
            
            self.final_data.append(new_product)


    def manage_mongo_data(self):
        """Funcția care actualizează baza de date."""
        for product in self.final_data:
            self.collection.update_one({"product_id":product["product_id"]}, {"$set":product}, upsert=True)


    def process_specification_full_names(self, product):
        """
        Procesarea numelor complete ale pieselor și returnarea scoruilor de performanță.

        :param product: produsul
        :type product: dict
        :return: scorurile de performanță
        :rtype: tuple
        """
        cpu_full_name, gpu_full_name = "", ""
        cpu_score, gpu_score = 0, 0
        
        try:
            cpu_manufacturer = product['specification_data']['Procesor']['Producator procesor']
            cpu_type = ""
            cpu_model = ""

            if 'Tip procesor' in product['specification_data']['Procesor']:
                cpu_type = product['specification_data']['Procesor']['Tip procesor']
            
            if 'Model procesor' in product['specification_data']['Procesor']:
                cpu_model = product['specification_data']['Procesor']['Model procesor']
            
            if cpu_manufacturer == "Apple":
                cpu_cores = f'{product["specification_data"]["Procesor"]["Numar nuclee"]} Core'
                cpu_full_name = f'{cpu_manufacturer} {cpu_type} {cpu_cores}'
            else:
                cpu_full_name = f"{cpu_manufacturer} {cpu_type} {cpu_model}"
        
        except KeyError:
            cpu_score = -1

        try:
            if "Tip placa video" in product['specification_data']["Placa video"]:
                if product['specification_data']["Placa video"]["Tip placa video"] == "Integrata":
                    if "Procesor grafic integrat" in product['specification_data']["Procesor"]:
                        gpu_full_name = product['specification_data']["Procesor"]["Procesor grafic integrat"]
                    else:
                        gpu_score = -1
                elif "Model placa video" in product['specification_data']["Placa video"]:
                    memory = ""
                    if "Capacitate memorie video" in product['specification_data']["Placa video"]:
                        memory_string = product['specification_data']["Placa video"]["Capacitate memorie video"]
                        memory_integer = 0
                        if memory_string.lower().endswith("mb"):
                            memory_integer = math.floor(int(memory_string[:-2])/1024)
                        memory = memory_string if memory_string.lower().endswith("gb") else f"{memory_integer}GB"

                    chipset_name = product['specification_data']['Placa video']['Chipset video']
                    if "nvidia" in chipset_name.lower():
                        chipset_name = chipset_name.lower().replace('nvidia ','').replace('-','').replace(' gtx','').replace('geforce','GeForce')
                    if "amd" in chipset_name.lower():
                        chipset_name = chipset_name.lower().replace('amd ','').replace('radeon','Radeon')
                    
                    gpu_model = product['specification_data']['Placa video']['Model placa video']

                    gpu_full_name = f"{chipset_name} {gpu_model} {memory}"
                else:
                    gpu_score = -1
            else:
                gpu_score = -1
        except KeyError:
            gpu_score = -1

        if gpu_score == 0:
            gpu_score = self.find_similar_gpu_or_cpu_score(gpu_full_name, 'gpu')
        if cpu_score == 0:
            cpu_score = self.find_similar_gpu_or_cpu_score(cpu_full_name, 'cpu')

        return cpu_score, gpu_score


    def find_similar_gpu_or_cpu_score(self, product_spec, type):
        """
        Extragerea scorului în funcție de numele piesei.

        :param product_spec: denumire piesei
        :type product_spec: str
        :param type: tipul piesei
        :type type: str
        :return: scorul
        :rtype: int
        """
        max_score = 0
        matching_key = None
        search_dictionary = self.extracted_cpu_scores if type == 'cpu' else self.extracted_gpu_scores
        
        for k in search_dictionary:
            score = fuzz.token_set_ratio(product_spec, k)
            if score > max_score:
                max_score = score
                matching_key = k

        if matching_key is not None:
            return search_dictionary[matching_key]
        else:
            return None

    def compute_mean_trending_review_count(self):
        """
        Determinarea indicelui de trend.

        :return: indicele de trend
        :rtype: int
        """
        maximum_review_count_for_six_months = []

        for product in self.final_data:
            reviews_dict_counter = {}

            for rev in product['review_data']:
                if datetime.datetime.strptime(rev['date_published'],"%Y-%m-%d").replace(day=1) in reviews_dict_counter:
                    reviews_dict_counter[datetime.datetime.strptime(rev['date_published'],"%Y-%m-%d").replace(day=1)] += 1
                else:
                    reviews_dict_counter[datetime.datetime.strptime(rev['date_published'],"%Y-%m-%d").replace(day=1)] = 1
                        
            reviews_dict_counter = dict(sorted(reviews_dict_counter.items()))
            review_counts = {}
            for i, date in enumerate(reviews_dict_counter):
                count = reviews_dict_counter[date]
                period_start = date
                period = f"{period_start.strftime('%b %Y')}"
                if period in review_counts:
                    review_counts[period] += count
                else:
                    review_counts[period] = count
            
            try:
                maximum_review_count_for_six_months.append(max(review_counts.values()))
            except ValueError:
                continue

        reviews_mean_per_six_months = floor(mean(sorted(maximum_review_count_for_six_months, reverse=True)[:10]))
        return reviews_mean_per_six_months

    def process_relevance_rating(self):
        """Calcularea scorului de relevanță."""       

        reviews_mean_per_six_months = self.compute_mean_trending_review_count()

        for i in range(len(self.final_data)):
            #Salvez datele despre recenzii intr-o lista de tipul [{datetime:sentiment_value}]
            review_dict_list = [{datetime.datetime.strptime(review["date_published"],"%Y-%m-%d").replace(day=1): review["sentiment"]} for review in self.final_data[i]['review_data']]
            review_dict_list = sorted(review_dict_list, key=lambda x: list(x.keys())[0])

            current_month = datetime.datetime.now().replace(day=1)
            six_months_ago = current_month - relativedelta(months=6)

            rating_six = 0
            count_six = 0
            for rev in review_dict_list:
                rev_date, rev_sentiment = next(iter(rev.items()))
                if six_months_ago <= rev_date <= current_month:
                    rating_six += rev_sentiment
                    count_six += 1
            
            trending_score = (0 if count_six==0 else rating_six / count_six)
            general_score = (0 if len(review_dict_list) == 0 else sum(next(iter(review.values())) for review in review_dict_list) / len(review_dict_list))
            relevance_score = 3 * general_score + trending_score + (1 if rating_six >= reviews_mean_per_six_months else rating_six/reviews_mean_per_six_months)
            self.final_data[i]['relevance_score'] = relevance_score
