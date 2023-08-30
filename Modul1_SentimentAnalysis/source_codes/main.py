import re
import os
import csv
import nltk
import string
import gensim
import numpy as np
import pandas as pd
import torch.nn as nn
import torch as torch
import seaborn as sns

from NeuralNetworks import *
from laroseda_datasets import LarosedaDatasetTrain, LarosedaDatasetTest

from tqdm import tqdm
from nltk.util import ngrams 
from collections import Counter
from nltk.corpus import stopwords
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

EPOCHS = 50
LEARNING_RATE = 1e-4 # AdamW
PLOTTING=True
LEN = 0
MEAN = 70
NGRAM = 1
STP = True
VOCAB_SIZE = 10000
USING_W2V = True
TRAIN_BATCH_SIZE = 16

stpword = stopwords.words('romanian')

class DataProcessor():
    def __init__(self):
        self.with_Word2Vec = False
        self.with_stopwords = False
        self.is_train_data = True

        self.mean = -1
        self.vocab_length = 0
        self.vocab_to_int = dict()
        self.word_list = dict()
        self.ngram = NGRAM

        self.w2vModel = None
        
    def set_stopwords(self, withStopwords):
        self.with_stopwords = withStopwords

    def set_with_Word2Vec(self, withWord2Vec):
        self.with_Word2Vec = withWord2Vec

    def set_is_train_data(self,isTrainData):
        self.is_train_data = isTrainData

    def set_mean(self,mean):
        self.mean = mean

    def set_vocab_length(self, vocab_length):
        self.vocab_length = vocab_length 
    
    def set_w2vModel(self, model):
        self.w2vModel = model
        self.with_Word2Vec = True
    
    def get_word_list_vocab(self):
        return self.word_list, self.vocab_to_int

    def get_vocab_length(self):
        if self.w2vModel:
            return self.w2vModel.wv.vectors.shape[1]
        else:
            return self.vocab_length

    def set_polarity_laroseda(self,dataframe):
        #sterg coloana "index" pentru ca e irelevanta
        dataframe = dataframe.drop(['index'], axis='columns')
        dataframe.dropna(inplace=True)
        dataframe['Polarity'] = dataframe['starRating'].apply(lambda x: 1 if x > 3 else 0)
        
        dataframe['Review'] = dataframe['content'] #.apply(self.get_text_processing)
        labels = dataframe.drop(["Review","title","starRating","content"], axis=1).values

        return dataframe, labels


    def get_text_processing(self, text):
        no_punctuation = [char for char in text if char not in string.punctuation]
        no_punctuation = ''.join(no_punctuation) #scoatem semnele de punctuatie
        
        if self.with_stopwords:
            return ' '.join([word.lower() for word in no_punctuation.split()])
        else:
            return ' '.join([word.lower() for word in no_punctuation.split() if word.lower() not in stpword])

    def remove_numbers(self,dataframe):
        regex = re.compile(r'\d+')

        remover = lambda x: regex.sub('',x)
        dataframe['Review'] = dataframe['Review'].apply(remover)

        for i in range(len(dataframe["Review"])):
            list_token = [token for token in dataframe['Review'][i].split()]
            dataframe['Review'][i] = " ".join(list_token)
            
        return dataframe

    def tokenize_dataset(self, dataframe, train):
        if self.with_Word2Vec:
            if train:
                tokenized_texts = [nltk.word_tokenize(text, language="english") for text in dataframe["Review"]]
            else:
                tokenized_texts = [nltk.word_tokenize(text, language="english") for text in dataframe["content"]]

            embeddings = []
            for tokens in tokenized_texts:
                text_embeddings = []
                for token in tokens:
                    if token.lower() in self.w2vModel.wv:
                        text_embeddings.append(self.w2vModel.wv[token.lower()])
                embeddings.append(text_embeddings)
            return embeddings

            # Varianta cu embeddings word2vec per propozitie, nu per cuvant

            # if train:
            #     tokenized_texts = [[token.lower() for token in sentence.split() if token.lower() in self.w2vModel.wv] for sentence in dataframe["Review"]]
            # else:
            #     tokenized_texts = [[token.lower() for token in sentence.split() if token.lower() in self.w2vModel.wv] for sentence in dataframe["content"]]

            # vectors = []
            # for tokens in tokenized_texts:
            #     if len(tokens) > 0:
            #         vectors.append(np.mean(self.w2vModel.wv[tokens], axis=0))
            #     # else:
            #         # vectors.append([])
            # return np.vstack(vectors).astype(float)
        else:
            review_col = "content"
            if train:
                review_col = "Review"
                one_huge_string = ' '.join(dataframe["Review"])
                all_words = list(one_huge_string.split())
                word_counts = Counter(ngrams(all_words, self.ngram))
                if self.vocab_length == 0:
                    self.vocab_length = len(word_counts)
                self.word_list = dict(word_counts.most_common(self.vocab_length))
                self.vocab_to_int = {word:idx for idx, word in enumerate(self.word_list)}

        
            encoded_reviews = []
            for review in dataframe[review_col]:
                encoded_reviews.append([])
                for bigram in ngrams(review.split(), self.ngram):
                    if bigram in self.word_list:
                        encoded_reviews[-1].append(self.vocab_to_int.get(bigram))
                   
            return encoded_reviews
    
    def padding(self,encoded_reviews, labels):
        # self.mean = round(sum(len(sublist) for sublist in encoded_reviews) / len(encoded_reviews))
        if self.with_Word2Vec:
            def pad_record(record):
                if len(record) < self.mean:
                    diff = self.mean - len(record)
                    record = np.pad(record,((0,diff),(0,0)), mode='constant')
                else:
                    record = record[:self.mean]
                return record
        else:
            def pad_record(review):
                if len(review) >= self.mean:
                    review = review[:self.mean]
                else:
                    review += [0]*(self.mean-len(review))
                return review
            
        processed_data = []
        labels_to_remove = []
        for i in range(len(encoded_reviews)):
            if len(encoded_reviews[i]) > 0:
                processed_data.append(pad_record(encoded_reviews[i]))
                # processed_data.append(encoded_reviews[i]) #tot pentru varianta cu word embedding per propozitie
            else:
                labels_to_remove.append(i)

        labels_to_remove.reverse()
        for i in labels_to_remove:
            labels = np.delete(labels,i)
        return np.array(processed_data), labels

    def preprocess_train_pipeline(self,dataframe):
        global MEAN
        dataframe, labels = self.set_polarity_laroseda(dataframe)
        dataframe = self.remove_numbers(dataframe)
        # max = 0
        # for sublist in dataframe["Review"]:
        #     if len(sublist.split()) > max :
        #         max = len(sublist.split())
        # self.mean = max
        # MEAN = max
        # if self.mean == -1:

        #     self.mean = round(sum(len(sublist.split()) for sublist in dataframe["Review"]) / len(dataframe["Review"]))
        #     MEAN = self.mean

        encoded_reviews = self.tokenize_dataset(dataframe, True)
        padded_data, labels = self.padding(encoded_reviews, labels)

        return padded_data, labels
    
    def preprocess_test_pipeline(self,dataframe):
        dataframe = dataframe.drop(['index'], axis='columns')
        dataframe.dropna(inplace=True)
        dataframe['Polarity'] = dataframe['starRating'].apply(lambda x: 1 if x > 3 else 0)
        labels = dataframe.drop(["title","starRating","content"], axis=1).values

        # Pentru calcularea mediei
        # if self.mean == -1:
            # self.mean = round(sum(len(sublist.split()) for sublist in dataframe["content"]) / len(dataframe["content"]))

        encoded_reviews = self.tokenize_dataset(dataframe, False)
        padded_data, labels = self.padding(encoded_reviews, labels)

        return padded_data, labels 




def train_epoch(model, train_dataloader, loss_crt, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    all_predictions = []
    all_labels = []
    num_batches = len(train_dataloader)

    for index,train_batch in tqdm(enumerate(train_dataloader)):
        model.parameters()
        batch_data, batch_labels = train_batch
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        raw_output = model(batch_data)

        batch_predictions = torch.argmax(raw_output, dim=1)
        loss = loss_crt(raw_output, batch_labels.squeeze(-1).long())
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_acc = accuracy_score(batch_labels.cpu().detach().numpy(), batch_predictions.cpu().detach().numpy())
        epoch_acc += batch_acc
        
        all_predictions += batch_predictions.tolist()
        all_labels += batch_labels.tolist()

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches
    
    return epoch_loss, epoch_acc, all_predictions, all_labels


def eval_epoch(model, eval_dataloader, loss_crt, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(eval_dataloader)
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for index, eval_batch in tqdm(enumerate(eval_dataloader)):
            batch_data, batch_labels = eval_batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            raw_output = model(batch_data)

            batch_predictions = torch.argmax(raw_output, dim=1)
            loss = loss_crt(raw_output, batch_labels.squeeze(-1).long())

            epoch_loss += loss.item()
            batch_acc = accuracy_score(batch_labels.cpu().detach().numpy(), batch_predictions.cpu().detach().numpy())
            epoch_acc += batch_acc

            all_predictions += batch_predictions.tolist()
            all_labels += batch_labels.tolist()
    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    return epoch_loss, epoch_acc, all_predictions, all_labels


def train(model, train_dataloader, val_dataloader, test_dataloader, device, train_batch_size):
    
    model.to(device)
    global LEN
    global MEAN
    global NGRAM
    global STP

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.937, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.937, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    loss_criterion = nn.CrossEntropyLoss()

    #fisier cu configuratia modelului
    new_path =''
    with open("../count.txt",'r+') as f:
       lines = f.readlines()
       new_dir = f'model{lines[0]}'
       new_path = f'{os.path.join("../saves/", new_dir)}'
       new_params = new_path + "/params.txt"
       os.mkdir(new_path)
       lines = int(lines[0]) + 1
       f.seek(0)
       f.write(str(lines))
    
       with open(new_params,'a+') as nf:
           model_params = str(model)
           nf.writelines(model_params)
           nf.writelines(optimizer.__str__())
           nf.writelines(scheduler.__str__())
           nf.writelines(loss_criterion.__str__())
           nf.writelines(f'\nW2V: {str(False) if model_params.find("Embedding") > -1 else str(True)}')
           nf.writelines(f"\nMean: {str(MEAN)}")
           nf.writelines(f"\nVocab length: {str(LEN)}")
           nf.writelines(f"\nN-gram: {str(NGRAM)}")
           nf.writelines(f"\nUsing stopwords: {str(STP)}")
           nf.writelines(f"\Batch size: {str(train_batch_size)}")
           nf.writelines(f"\Epochs : {str(EPOCHS)}")

    print("Training:")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    max_eval = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy, train_pred_ep, train_labels_ep = train_epoch(
            model, train_dataloader, loss_criterion, optimizer, device)

        val_loss, val_accuracy, val_pred_ep, val_labels_ep = eval_epoch(model, val_dataloader, loss_criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print('\nEpoch %d' % (epoch))
        print('train loss: %10.8f, accuracy: %10.8f' %
                (train_loss, train_accuracy))
        print('val loss: %10.8f, accuracy: %10.8f\n' %
                (val_loss, val_accuracy))
        if val_accuracy > max_eval:
            print('Checkpoint saved: %10.8f > %10.8f\n' %
                  (val_accuracy,max_eval))
            max_eval = val_accuracy
            torch.save(model.state_dict(), f'{new_path}/checkpoint_acc{str(int(val_accuracy*1000000))}.pt')

    torch.save(model.state_dict(), f'{new_path}/final_model{str(int(val_accuracy*1000000))}.pt')
    val_loss, val_accuracy, val_pred_ep, val_labels_ep = eval_epoch(model, val_dataloader, loss_criterion, device)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    if PLOTTING:
        fig, axis = plt.subplots(2,2, figsize=(10,10))
        epochs_list = range(EPOCHS)

        axis[0,0].plot(epochs_list, train_losses)
        axis[0,0].set_title("Train losses")
        axis[0,1].plot(epochs_list, val_losses)
        axis[0,1].set_title("Val losses")
        axis[1,0].plot(epochs_list, train_accuracies)
        axis[1,0].set_title("Train accuracy")
        axis[1,1].plot(epochs_list, val_accuracies)
        axis[1,1].set_title("Val accuracy")

        axis[1,1].set(xlabel='Epochs', ylabel="")
        axis[1,0].set(xlabel='Epochs', ylabel="Accuracy")

        plt.savefig(f'{new_path}/final_model_plots.png')
        plt.close()

        val_confusion_matrix = confusion_matrix(val_labels_ep, val_pred_ep)
        val_confusion_matrix_df = pd.DataFrame(val_confusion_matrix, range(2), range(2))
        sns.set(font_scale=1.1)
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(val_confusion_matrix_df, annot=True, annot_kws={"size":12}, fmt='d')

        plt.savefig(f'{new_path}/final_model_confusion_matrix.png')
        plt.close()


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])

    dp = DataProcessor()
    dp.set_stopwords(STP)
    dp.set_mean(MEAN)
    if USING_W2V:
        print("Loading w2v...")
        dp.set_w2vModel(gensim.models.Word2Vec.load("../word2vec_models/w2v_byme.model"))
        # Varianta veche de model word2vec
        # dp.set_w2vModel(gensim.models.keyedvectors.KeyedVectors.load_word2vec_format("model.bin", binary=True))
        VOCAB_SIZE = dp.get_vocab_length()
    else:
        # dp.setVocabLength(vocab_size)
        pass

    print("Preprocessing train data...")
    encoded_train_data, train_labels = dp.preprocess_train_pipeline(train_dataframe)
    VOCAB_SIZE = dp.get_vocab_length()
    train_x, train_y = encoded_train_data, train_labels

    print("Preprocessing eval data...")
    encoded_eval_data, eval_labels = dp.preprocess_test_pipeline(test_dataframe)
    valid_x, valid_y = encoded_eval_data, eval_labels

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, eval_dataset, test_dataset = None, None, None
    
    if USING_W2V:
        train_dataset    = LarosedaDatasetTrain(train_x, train_y)
        eval_dataset     = LarosedaDatasetTrain(valid_x, valid_y)
        test_dataset     = LarosedaDatasetTest(valid_x)
    else:
        train_dataset    = LarosedaDatasetTrain(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
        eval_dataset     = LarosedaDatasetTrain(torch.from_numpy(valid_x).to(device), torch.from_numpy(valid_y).to(device))
        test_dataset     = LarosedaDatasetTest(torch.from_numpy(valid_x).to(device))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                num_workers=0,
                                                shuffle=True,
                                                drop_last=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                batch_size=15,
                                                num_workers=0,
                                                shuffle=False,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=15,
                                                num_workers=0,
                                                shuffle=False,
                                                drop_last=False)

    # model = CNN(VOCAB_SIZE, 128, [15,19,23], 128, 2, 0.2)
    # model = LSTM_1L(VOCAB_SIZE)
    model = BiLSTM_W2v(VOCAB_SIZE)

    train(model, train_loader, eval_loader, test_loader, device, 15)
