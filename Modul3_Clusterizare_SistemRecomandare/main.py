"""Punctul de pornire al serverului."""
#================================ IMPORTS =================================
import base64

from cluster_plot import Plot
from preprocessing import DataProcessor
from clustering import Cluster
import numpy as np

from flask_cors import CORS
from flask import Flask, jsonify, request, abort
from recommendation import Recommendation
from mongo_processing import MongoProcessing

#================================= MONGO ==================================
mongo_client_address = "mongodb://admin:passwdadmin@192.168.56.20:27017"
database_name= "EmagProductDatabase"
app = Flask(__name__)
CORS(app)
mongo_client = MongoProcessing()
recommendation_obj = Recommendation()

# ================================= ENDPOINTS ==============================
@app.route('/laptops/<string:laptop_url>')
def get_laptop(laptop_url):
    """
    Extragerea informatiilor despre un anumit laptop.

    :param laptop_url: Link-ul laptopului
    :type laptop_url: str
    :return: JSON cu datele despre laptop
    :rtype: JSON
    """
    laptop = mongo_client.get_laptop(laptop_url)
    if laptop is None:
        abort(404)
    return jsonify(laptop)

@app.route('/laptops')
def get_all_laptops():
    laptop_collection = mongo_client.get_all_laptops()
    return jsonify(laptop_collection)

@app.route('/recluster', methods=['POST'])
def redo_clustering():
    """
    Refacerea clusterizării

    :return: mesaj, HTTP status code
    :rtype: string, int
    """
    global recommendation_obj
    try:
        new_recommendation_obj = Recommendation()
        recommendation_obj = new_recommendation_obj
        return "", 200
    except Exception as e:
        return "", 500

@app.route('/laptops/<string:laptop_link>/recommendations', methods=['GET'])
def get_recommendation_for_laptop(laptop_link):
    """
    Obținerea de recomandări și alternative similare

    :param laptop_url: Link-ul laptopului
    :type laptop_url: str
    :return: JSON cu datele despre laptop
    :rtype: JSON
    """
    try:
        laptop_link = base64.b64decode(laptop_link+ "=").decode('utf-8') 
        min_price = float(request.args.get('min', 0))
        max_price = float(request.args.get('max', float('inf')))

        laptop_collection = mongo_client.get_all_laptop_data()
        recommendation_links = recommendation_obj.get_recommendation_links(laptop_link, min_price, max_price)
        final_result = {"similar": [], "recommended": []}

        for dict in laptop_collection:
            url = dict.get("product_url")

            for key, value in recommendation_links.items():
                if url in value:                
                    final_result[key].append(dict)
                    recommendation_links[key].remove(url)

            if len(recommendation_links['similar']) == 0 and len(recommendation_links['recommended']) == 0:
                break
            
        return jsonify(final_result)
    except Exception as e:
        return "", 500

if __name__ == '__main__':
    app.run(debug=False)
