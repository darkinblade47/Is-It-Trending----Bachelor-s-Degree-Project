import re
import json
import base64
import scrapy
import psycopg2

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC

with open('scrapy_project/paths/emag.json') as file:
    product_xpaths = json.load(file)['laptops']

with open('scrapy_project/paths/scores.json') as file:
    score_xpaths = json.load(file)

class EmagLaptopSpider(scrapy.Spider):
    """
    Spider de extragere a datelor despre laptopurile din cadrul eMag.

    :param scrapy: spider
    :type scrapy: spider
    :yield: items
    :rtype: object
    """
    name = "EmagLaptopSpider"
    start_urls  = product_xpaths["start_urls"]
    scores_urls  = product_xpaths["scores_urls"]

    count_s = 0
    page_count = 1
    scores_crawled = False

    def parse(self, response):
        """
        Funcția care se apelează la începutul procesului de scraping și după fiecare pagină completă de produse.

        :param response: răspunsul primit
        :type response: requests.Response
        :yield: dicționare cu informații
        :rtype: dict
        """
        if not self.scores_crawled:
            self.scores_crawled = True
            score_generator = self.scores_crawl()
            score_response = next(score_generator)
            cpu_scores, gpu_scores = score_response['cpu_score'], score_response['gpu_score']
            yield {"gpu_score": gpu_scores, "cpu_score": cpu_scores}

        product_urls_selectors = response.xpath(product_xpaths["laptop_urls_xpath"]).xpath('@href')
        product_urls = [url.get() for url in product_urls_selectors]
        for url in product_urls:
            domain = url[:20]
            product = url[20:]
            part_key = product.split('/')[-2]
            if product.endswith("#used-products"): # fara resigilate
                continue
            
            yield scrapy.Request(url, callback = self.parse_laptop, dont_filter=True, meta={'product_url':url})
            api_review_url = f"{domain}product-feedback/{product}reviews/list"
            api_review_url_limit = f"{api_review_url}?page%5Blimit%5D=100&page%5Boffset%5D=0"
            yield scrapy.Request(api_review_url_limit, callback = self.parse_laptop_reviews, meta={'api_url': api_review_url, 'offset':0, 'part_key':part_key}, dont_filter=True)

        next_page_li = response.xpath(product_xpaths["next_page_li_xpath"])
        if product_xpaths["disabled_tag"] not in next_page_li.xpath('@class').get(default=''):
            self.page_count+=1                             
            next_page_of_products = f"https://www.emag.ro/laptopuri/sort-reviewsdesc/p{self.page_count}/c"
            yield scrapy.Request(url = next_page_of_products, callback = self.parse)

    def parse_laptop(self, response):
        """
        Funcția care se apelează pentru extragerea specificațiilor despre laptop.

        :param response: răspunsul primit
        :type response: requests.Response
        :yield: dicționare cu informații
        :rtype: dict
        """
        self.count_s+=1
        print(f'Laptopuri extrase: {self.count_s}')
        product_dict = {}
        product_individual_id = None 
        stock_status = -1
        try:
            stock_paragraph = response.xpath(product_xpaths["stock_paragraph_xpath"])
            first_span = stock_paragraph.xpath(product_xpaths["stock_label_xpath"])
            if product_xpaths["limited_stock"] in first_span.xpath("@class").get(default=''):
                stock_status = 0
            elif product_xpaths["in_stock"] in first_span.xpath("@class").get(default=''):
                stock_status = 1
        except Exception:
            stock_status = -1
                 
        # Tag-urile tbody ce contin o grupare de specificatii. Ex: pentru procesoare: producator, tip, model etc
        specifications_tb = response.xpath(product_xpaths["specifications_table_tbody_xpath"])
        # Tag-urile p ce contin denumirea grupului de specificatii. Ex: procesor, afisare, memorie, etc
        specifications_p = response.xpath(product_xpaths["specifications_table_paragraphs_xpath"]).getall()

        javascripts = response.xpath(product_xpaths["js_scripts_xpath"]).getall()[0]
        laptop_model = response.xpath(product_xpaths["laptop_model_xpath"]).get()[:-10]
        product_data_script = response.xpath(product_xpaths["js_script_xpath"])[1].get()

        brand_match = re.search(r'"brand":\s*"([^"]+)"', javascripts)
        brand = ""
        if brand_match:
            brand = brand_match.group(1)
        
        price_match = re.search(r'"price":\s([\d.]+)', javascripts)
        price = ""
        if brand_match:
            price = price_match.group(1)

        prod_inv_id_match = re.search('EM.product_id = (\d+);', product_data_script)
        if prod_inv_id_match:
            product_individual_id = str(prod_inv_id_match.group(1))

        for i in range(len(specifications_tb)):
            list_of_subspecifications = {}
            for tr in specifications_tb[i].xpath('./tr'):
                row_key = tr.xpath(product_xpaths["table_row_key_xpath"]).get() 
                row_value = tr.xpath(product_xpaths["table_row_value_xpath"]).get()[:-1]
                list_of_subspecifications[row_key] = row_value

            product_dict.update({specifications_p[i]: list_of_subspecifications})

        laptop_image = response.xpath(product_xpaths["product_image_xpath"]).get() 
        yield {"specifications":{"product_id":product_individual_id, "product_url":response.meta['product_url'], "image":laptop_image, "brand":brand, "laptop_model":laptop_model, "price":float(price), "specification_data":product_dict, "stock":stock_status}}

    def parse_laptop_reviews(self, response):
        """
        Funcția care se apelează pentru extragerea recenziilor despre laptop.

        :param response: răspunsul primit
        :type response: requests.Response
        :yield: dicționare cu informații
        :rtype: dict
        """
        data = json.loads(response.body)
        review_count = data['reviews']['count']
        if review_count==0:
            return
        
        reviews = data['reviews']['items']
        reviews_dict = {}
        product_id = None

        for review in reviews:
            if review['is_bought']==False:
                continue

            if review['product']['part_number_key'] == response.meta['part_key']:

                if product_id is None:
                    product_id = review['product']['id']
                
                review_dict = {
                    'title' : review['title'],
                    'review_content' : review['content_no_tags'],
                    'date_published' : review['published'][:10]
                }

                if product_id in reviews_dict:
                    reviews_dict[product_id].append(review_dict)
                else:
                    reviews_dict[product_id] = [review_dict]

        extracted_data = {'reviews':reviews_dict}
        yield {'extracted_reviews': extracted_data}


        offset = response.meta['offset'] + 100 
        if offset < review_count:
            api_url = response.meta['api_url']
            api_url+= f'?page%5Blimit%5D=100&page%5Boffset%5D={offset}'

            yield scrapy.Request(api_url, callback = self.parse_laptop_reviews, meta={'api_url': response.meta['api_url'], 'offset':offset, 'extracted_reviews':extracted_data, 'part_key':response.meta['part_key']}, dont_filter=True)        

    def scores_crawl(self):
        """
        Funcția de extragere a scorurilor de perfomanță utilizând Selenium.

        :yield: dicționar cu scoruri
        :rtype: dict
        """
        forbidden_list = ['(Mobile)', 'Mobile', 'Laptop', '(Laptop)', 'GPU', 'With', 'with', 'Design']

        list_of_gpus = {}
        list_of_cpus = {}
        extract_tries = 5

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--start-maximized')

        try:
            while extract_tries >= 0:
                driver = webdriver.Chrome(options=options)

                for index, url in enumerate(self.scores_urls):
                    driver.get(url)
                    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, score_xpaths["cookie_confirm_xpath"]))).click()

                    category_select_tag = driver.find_element(By.XPATH,score_xpaths["category_select_xpath"])
                    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH,score_xpaths["category_select_xpath"])))
                    select_elem = Select(category_select_tag)
                    select_elem.select_by_visible_text('Mobile' if index == 0 else 'Laptop')
                    
                    show_all_results_select = driver.find_element(By.XPATH, score_xpaths["all_results_select_xpath"])
                    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, score_xpaths["all_results_select_xpath"]))) #.select_by_visible_text('All')
                    results_select = Select(show_all_results_select)
                    results_select.select_by_visible_text('All')

                    table = driver.find_elements(By.XPATH, score_xpaths["score_table_xpath"])
                    if index == 0:
                        for i in range(len(table)):
                            gpu_name = table[i].find_elements(By.XPATH, "./td")[1].text
                            pattern = r'\b(?:{})\b'.format('|'.join(forbidden_list)) #filtrez cuvintele irelevante din numele placii video
                            gpu_name = re.sub(pattern, '', gpu_name, flags=re.IGNORECASE).rstrip()
                            if gpu_name == "GeForce":
                                continue
                            gpu_score = table[i].find_elements(By.XPATH, "./td")[2].text
                            list_of_gpus[gpu_name] = int(gpu_score.replace(',', ''))
                    else:
                        for i in range(len(table)):
                            cpu_name = table[i].find_elements(By.XPATH, "./td")[1].text
                            cpu_score = table[i].find_elements(By.XPATH, "./td")[3].text
                            list_of_cpus[cpu_name] = int(cpu_score.replace(',', ''))
                
                extract_tries = -1  
                driver.quit()
        except Exception:
            driver.quit()
            extract_tries -= 1

        yield {'cpu_score': list_of_cpus, 'gpu_score': list_of_gpus}
