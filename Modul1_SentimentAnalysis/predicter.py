import os
import torch
import gensim
import numpy as np
from tqdm import tqdm
from source_codes.NeuralNetworks import BiLSTM_W2v
from nltk import word_tokenize

FOLDER_PATH = os.path.dirname(__file__)

class SentimentPredict():
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.w2vModel = gensim.models.Word2Vec.load(os.path.join(FOLDER_PATH,"word2vec_models/w2v_byme.model"))
        self.model = BiLSTM_W2v(self.w2vModel.wv.vectors.shape[1])
        self.model.load_state_dict(torch.load(os.path.join(FOLDER_PATH, "saves\model306_9201\checkpoint_acc920120.pt")))
    
    def pad_record(self, record):
        self.mean = len(record)
        if len(record) < self.mean:
            diff = self.mean - len(record)
            record = np.pad(record,((0,diff),(0,0)), mode='constant')
        else:
            record = record[:self.mean]
        return record

    def tokenize_and_encode(self, reviews):
        tokenized_texts = [word_tokenize(text, language="english") for text in reviews]

        embeddings = []
        for tokens in tokenized_texts:
            text_embeddings = []
            for token in tokens:
                if token.lower() in self.w2vModel.wv:
                    text_embeddings.append(self.w2vModel.wv[token.lower()])
            embeddings.append(text_embeddings)
        return embeddings
    
    def padding(self, encoded_reviews):
        processed_data = []
        for i in range(len(encoded_reviews)):
            if len(encoded_reviews[i]) > 0:
                processed_data.append(np.array(self.pad_record(encoded_reviews[i])))
            else:
                processed_data.append("NO CONTENT")

        return processed_data
    
    def preprocess_data_for_predict(self, reviews):
        encoded_reviews = self.tokenize_and_encode(reviews)
        final_data = self.padding(encoded_reviews)
        for i in range(len(final_data)):
            if final_data[i] != "NO CONTENT":
                data = final_data[i]
                final_data[i] = torch.from_numpy(data.reshape(1, data.shape[0], data.shape[1]))

        return final_data
    
    def predict(self, review_list):
        data_to_be_evaluated = self.preprocess_data_for_predict(review_list)
                
        self.model.to(self.device)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            try:
                for idx, batch in enumerate(data_to_be_evaluated):
                    if batch == "NO CONTENT":
                        predictions += [-1]
                        continue

                    batch_data = batch.to(self.device)

                    raw_output = self.model(batch_data)

                    batch_predictions = torch.argmax(raw_output, dim=1)

                    predictions += batch_predictions.tolist()
            except Exception as e:
                predictions += [-1]
        return predictions


