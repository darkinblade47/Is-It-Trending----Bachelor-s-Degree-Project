import axios from "axios";

export default class MongoService {
  constructor(link) {
    this.http = axios.create({
      headers: {
        "Content-type": "application/json",
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept'
      }
    });
    this.link = btoa(link).replace("=","");
  }

  async getLaptop() {
    let res = null
    try{
      const response = await this.http.get(`http://localhost:5000/laptops/${this.link}`)
      res = {"valid":response.data}
    }
    catch (error) {
      if (error.response.status == 404){
        
        return {"error": 404}
      }
    }

    return res
  }

  async getRecommendation(minPrice, maxPrice){
    let res = null
    
    try{
      const response = await this.http.get(`http://localhost:5000/laptops/${this.link}/recommendations?min=${minPrice}&max=${maxPrice}`)
      res = response.data
    }
    catch (error) {
    }

    return res
  }

}
