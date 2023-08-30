import base64
from pymongo import MongoClient

class MongoProcessing():
    def __init__(self) -> None:
        self.client = MongoClient("mongodb://admin:passwdadmin@192.168.56.20:27017")
        self.db = self.client["EmagProductDatabase"]
        
    def get_laptop(self, link):

        decoded_link = base64.b64decode(link+ "=").decode('utf-8') 
        query_result = self.db.emag_scrapy.find_one({'product_url':decoded_link})
        if query_result is None:
            return None
            
        laptop_data = {k: v for k, v in query_result.items() if k != "_id"}
        return laptop_data
    
    def get_all_laptops(self):
        query_result = self.db.emag_scrapy.find({})
        response = {k: v for k, v in query_result.items() if k == "price" and k=="relevance_score" and k=="scores"}
        
        return response

    def get_all_laptop_data(self):
        query_result = self.db.emag_scrapy.find({})
        
        data_list = []
        for document in query_result:
            document.pop('_id', None) 
            document.pop('review_data', None)
            document.pop('scores', None)
            data_list.append(document)

        return data_list

