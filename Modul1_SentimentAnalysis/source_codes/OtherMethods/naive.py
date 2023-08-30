#Credit: https://www.analyticsvidhya.com/blog/2022/03/building-naive-bayes-classifier-from-scratch-to-perform-sentiment-analysis/

import math
import re
import spacy
import nltk
import string
import numpy as np
import pandas as pd
import gensim

from sklearn.preprocessing import MinMaxScaler
from keras_preprocessing.sequence import pad_sequences
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from collections import defaultdict
from datasets import load_dataset
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

withStopwords = True

def set_polarity_laroseda(dataframe):
    dataframe = dataframe.drop(['index'], axis='columns')
    dataframe.dropna(inplace=True)
    dataframe['Polarity'] = dataframe['starRating'].apply(lambda x: 1 if x > 3 else 0)

    dataframe['Review'] = dataframe['content'].apply(get_text_processing)
    labels = dataframe.drop(["Review","title","starRating","content"], axis=1).values
    return dataframe, labels

def get_text_processing(text):
    global withStopwords
    stpword = stopwords.words('romanian')
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    if withStopwords:
        return ' '.join([word.lower() for word in no_punctuation.split()])
    else:
        stpwrd =  ' '.join([word.lower() for word in no_punctuation.split() if word.lower() not in stpword])
        return stpwrd

def remove_numbers(dataframe):
    regex = re.compile(r'\d+')
    remover = lambda x: regex.sub('',x)
    dataframe['Review'] = dataframe['Review'].apply(remover)
    for i in range(len(dataframe["Review"])):
        list_token = [token for token in dataframe['Review'][i].split()]
        dataframe['Review'][i] = " ".join(list_token)    
    return dataframe

def lemmatize(dataframe):
    lm_obj = spacy.load('ro_core_news_sm')
    for i in range(len(dataframe['Review'])):
        review_doc = lm_obj(np.array(dataframe['Review'])[i])
        dataframe['Review'][i] = " ".join(token.lemma_ for token in review_doc)


    return dataframe
    
def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)

def group_by_label(x, y, labels):
    data = {}

    # for l in labels:
    indexes_for_negatives = np.where(y == 0)[0]
    indexes_for_positives = np.where(y == 1)[0]
    data[0] = x[indexes_for_negatives]
    data[1] = x[indexes_for_positives]
    return data

def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors

def preprocess_train_pipeline(dataframe):
    dataframe, labels = set_polarity_laroseda(dataframe)
    dataframe = remove_numbers(dataframe)
    # dataframe = lemmatize(dataframe)
    return dataframe, labels


def main():
    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])

    #============================ RUNNING FLAGS ==========================
    global MEAN
    global LEN
    global NGRAM
    global STP
    MEAN = 50
    NGRAM = 1
    STP = True
    vocab_size = 5000
    w2Flag = True
    train_batch_size = 16
    #=====================================================================

    dp = DataPreprocessor()
    dp.useStopwords(STP)
    # dp.setMean(MEAN)
    if w2Flag:
        print("Loading w2v...")
        dp.set_w2vModel(gensim.models.keyedvectors.KeyedVectors.load_word2vec_format("model.bin", binary=True))
        vocab_size = dp.getVocabLength()
    else:
        pass
        dp.setVocabLength(vocab_size)
    print("Preprocessing train data...")
    encoded_train_data, train_labels = dp.preprocess_train_pipeline(train_dataframe)
    vocab_size = dp.getVocabLength()
    train_x, train_y = encoded_train_data, train_labels
    print("Preprocessing eval data...")

    encoded_eval_data, eval_labels = dp.preprocess_test_pipeline(test_dataframe)
    valid_x, valid_y = encoded_eval_data, eval_labels
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_datass, eval_datass, test_datass = None, None, None
    
    if w2Flag:
        train_datass    = LarosedaDatasetTrain(train_x, train_y)
        eval_datass     = LarosedaDatasetTrain(valid_x, valid_y)
        test_datass     = LarosedaDatasetTest(valid_x)
    else:
        train_datass    = LarosedaDatasetTrain(torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device))
        eval_datass     = LarosedaDatasetTrain(torch.from_numpy(valid_x).to(device), torch.from_numpy(valid_y).to(device))
        test_datass     = LarosedaDatasetTest(torch.from_numpy(valid_x).to(device))
    train_loader = torch.utils.data.DataLoader(dataset=train_datass,
                                                batch_size=train_batch_size,
                                                num_workers=0,
                                                shuffle=True,
                                                drop_last=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_datass,
                                                batch_size=4,
                                                num_workers=0,
                                                shuffle=False,
                                                drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_datass,
                                                batch_size=4,
                                                num_workers=0,
                                                shuffle=False,
                                                drop_last=True)
    LEN = vocab_size
    model = LSTM_W2v(vocab_size)
    train(model, train_loader, eval_loader, test_loader, device, train_batch_size)

def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    result = []
    for text in x:
        label_scores = {l[0]: log_label_priors[l[0]] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l[0])
                label_scores[l[0]] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result

def NB_integer():
    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    countt = 0
    
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    dataframe, labels = preprocess_train_pipeline(train_dataframe)
    
    vec = CountVectorizer()
    
    X = vec.fit_transform(np.array(dataframe['Review']))
    
    vocab = vec.get_feature_names_out()
    
    X = X.toarray()
    
    word_counts = {}
    for j in range(2):
        word_counts[j] = defaultdict(lambda: 0)
    
    for i in range(X.shape[0]):
        j = labels[i][0]
        for k in range(len(vocab)):
            word_counts[j][vocab[k]] += X[i][k]
    
    n_label_items, log_label_priors = fit(np.array(dataframe['Review']),labels,[0,1])
    test_dataframe, test_labels = preprocess_train_pipeline(test_dataframe)
    pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_dataframe['Review'])
    print("Accuracy of prediction on test set : ", accuracy_score(test_labels,pred))
    #0.794, with stopwords full
    #0.8396666666666667 without stopwords full

def NB_word2vec_2():
    def encode_data(data):
        encoded_data = []
        for sentence in data:
            sentence_vector = []
            for word in sentence.split():
                try:
                    word_vector = model.wv[word.lower()]
                    sentence_vector.append(word_vector)
                except KeyError:
                    pass
            encoded_data.append(sentence_vector)
        return encoded_data


    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    countt = 0
    
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    dataframe, labels = preprocess_train_pipeline(train_dataframe)
    test_dataframe, test_labels = preprocess_train_pipeline(test_dataframe)
    model = gensim.models.Word2Vec.load("w2v_byme.model")

    max_len = max(len(sentence) for sentence in encode_data(dataframe['Review']))

    # min_value_train = min([min(sentence) for sentence in np.array(encode_data(dataframe['Review']))])
    # min_value_eval = min([min(sentence) for sentence in np.array(encode_data(test_dataframe['Review']))])
    X_train_encoded = encode_data(dataframe['Review']).reshape((12000, -1, 100))
    X_eval_encoded = encode_data(test_dataframe['Review']).reshape((12000, -1, 100))

    min_value_train  = np.min(encode_data(dataframe['Review']), axis =(0,1,2))
    min_value_eval = np.min(encode_data(test_dataframe['Review']), axis =(0,1,2))

    X_train_encoded = np.array(encode_data(dataframe['Review']))
    X_eval_encoded = np.array(encode_data(test_dataframe['Review']))
    X_train_encoded_shifted = [[vec + abs(min_value_train) for vec in sentence] for sentence in X_train_encoded]
    X_eval_encoded_shifted = [[vec + abs(min_value_eval) for vec in sentence] for sentence in X_eval_encoded]
    X_train_encoded = pad_sequences(X_train_encoded_shifted, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0)
    X_eval_encoded = pad_sequences(X_eval_encoded_shifted, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0)

    # Convert encoded data to numpy arrays
    X_train_encoded = np.array(X_train_encoded)
    X_eval_encoded = np.array(X_eval_encoded)
    
    X_train_encoded = X_train_encoded.reshape((len(X_train_encoded), -1))
    X_eval_encoded = X_eval_encoded.reshape((len(X_eval_encoded), -1))

    clf = MultinomialNB()
    clf.fit(X_train_encoded, np.array(dataframe['Polarity']))

    accuracy = clf.score(X_eval_encoded, np.array(test_dataframe['Polarity']))
    print('Accuracy:', accuracy)

def NB_word2vec_2():

    def shift_features(X):
        # Find the minimum feature value
        min_feature = np.min(X)
        # If minimum feature value is negative, shift all features by its absolute value
        if min_feature < 0:
            X = X + abs(min_feature)
        return X

    
    # Define training data
    X_train = []
    y_train = []

    # Define test data
    X_test = []
    y_test = []

    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    countt = 0
    
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    dataframe, labels = preprocess_train_pipeline(train_dataframe)
    test_dataframe, test_labels = preprocess_train_pipeline(test_dataframe)
    model = gensim.models.Word2Vec.load("w2v_byme.model")

    for index, text in enumerate(dataframe['Review']):
    # Convert each word to a 100-dimensional vector using the Word2Vec model
        label = labels[index]
        text_vec = [model.wv[word.lower()] for word in text if word.lower() in model.wv]
        # Convert the list of vectors to a single vector by taking the average of all vectors
        text_vec = np.mean(text_vec, axis=0)
        # Shift negative features
        text_vec = shift_features(text_vec)
        # Add the vector to the training data
        X_train.append(text_vec)
        y_train.append(label)

    for index,text in enumerate(test_dataframe['Review']):
        label = test_labels[index]

        text_vec = [model.wv[word.lower()] for word in text if word.lower() in model.wv]
        text_vec = np.mean(text_vec, axis=0)
        text_vec = shift_features(text_vec)
        X_test.append(text_vec)
        y_test.append(label)
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

def NB_word2vec_3():

    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    dataframe, labels = preprocess_train_pipeline(train_dataframe)
    test_dataframe, test_labels = preprocess_train_pipeline(test_dataframe)
    model = gensim.models.Word2Vec.load("w2v_byme.model")

    def doc_vector(doc):
        # Split document into words
        words = doc.split()
        # Initialize document vector
        doc_vec = np.zeros(100)
        # Loop through words and add to document vector
        for word in words:
            try:
                # Get word vector from model
                word_vec = model.wv[word.lower()]
                # Shift negative values
                idx = word_vec < 0
                word_vec[idx] += 1
                # Add to document vector
                doc_vec += word_vec
            except KeyError:
                # Ignore words not in model
                pass
        return doc_vec

    # Define function to train Naive Bayes classifier
    def train_nb(X_train, y_train):
        # Get number of training examples and number of features
        m, n = X_train.shape
        # Initialize parameters
        phi_y = np.mean(y_train)
        phi_y_given_x = np.zeros((2, n))
        # Loop through features
        for j in range(n):
            # Get feature values for positive and negative classes
            X_j_pos = X_train[y_train == 1, j]
            X_j_neg = X_train[y_train == 0, j]
            # Calculate means and variances for positive and negative classes
            mu_pos, var_pos = np.mean(X_j_pos), np.var(X_j_pos)
            mu_neg, var_neg = np.mean(X_j_neg), np.var(X_j_neg)
            # Calculate parameters for Naive Bayes classifier
            phi_y_given_x[1, j] = (np.sum(X_j_pos) + mu_pos) / (np.sum(X_j_pos) + np.sum(X_j_neg) + mu_pos + mu_neg)
            phi_y_given_x[0, j] = (np.sum(X_j_neg) + mu_neg) / (np.sum(X_j_pos) + np.sum(X_j_neg) + mu_pos + mu_neg)
        return phi_y, phi_y_given_x

    # Define function to predict class for test examples
    def predict_nb(X_test, phi_y, phi_y_given_x):
        # Get number of test examples and number of features
        m, n = X_test.shape
        # Initialize predicted classes
        y_pred = np.zeros(m)
        # Loop through test examples
        for i in range(m):
            # Initialize log probabilities for positive and negative classes
            log_p_pos = np.log(phi_y_given_x[1])
            log_p_neg = np.log(phi_y_given_x[0])
            # Loop through features and calculate log probabilities
            for j in range(n):
                x_ij = X_test[i, j]
                mu_pos, var_pos = phi_y_given_x[1, j], np.var(X_train[y_train == 1, j])
                mu_neg, var_neg = phi_y_given_x[0, j], np.var(X_train[y_train == 0, j])
                log_p_neg[j] += np.log((1 / np.sqrt(2 * np.pi * var_neg)) * np.exp(-((x_ij - mu_neg) ** 2) / (2 * var_neg)))
        # Calculate total log probabilities for positive and negative classes
            log_p_total_pos = np.sum(log_p_pos) + np.log(phi_y)
            log_p_total_neg = np.sum(log_p_neg) + np.log(1 - phi_y)
            # Predict class based on total log probabilities
            if log_p_total_pos > log_p_total_neg:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

    X_train = []
    y_train = []

    for doc, label in zip(np.array(dataframe['Review']), labels):
        doc_vec = doc_vector(doc)
        X_train.append(doc_vec)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    phi_y, phi_y_given_x = train_nb(X_train, y_train)

    X_test = []
    y_test = []

    for doc, label in zip(np.array(test_dataframe['Review']), test_labels):
        doc_vec = doc_vector(doc)
        X_test.append(doc_vec)
        y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = predict_nb(X_test, phi_y, phi_y_given_x)

    accuracy = np.mean(y_pred == y_test)
    print('Accuracy:', accuracy)

def NB_word2vec_4():
    print("Loading dataset...")
    dataset = load_dataset("laroseda")
    
    train_dataframe, test_dataframe = pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    # dataframe, labels = preprocess_train_pipeline(train_dataframe)
    dataframe, labels = set_polarity_laroseda(train_dataframe)
    test_dataframe, test_labels = set_polarity_laroseda(test_dataframe)
    # test_dataframe, test_labels = preprocess_train_pipeline(test_dataframe)
    model = gensim.models.Word2Vec.load("w2v_byme.model")

    word_vectors = model.wv
    def text_to_vectors(text):
        vectors = []
        for word in text.split():
            if word in model.wv:
                vectors.append(model.wv[word])
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Load data
    train_texts = np.array(dataframe['Review'])
    train_labels = labels

    test_texts = np.array(test_dataframe['Review'])
    test_labels = test_labels
    # Define a pipeline for the classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word', tokenizer=lambda x: x.split(), lowercase=True)),
        ('classifier', MultinomialNB())
    ])
    scaler = MinMaxScaler()

    # Fit the pipeline on the training data
    pipeline.fit(train_texts, train_labels)

    # Evaluate the performance on the test data
    predicted_labels = pipeline.predict(test_texts)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")

    # Use Word2Vec Naive Bayes for sentiment analysis
    train_vectors = np.array([text_to_vectors(text) for text in train_texts])
    test_vectors = np.array([text_to_vectors(text) for text in test_texts])
    classifier = GaussianNB()
    classifier.fit(train_vectors, train_labels)
    predicted_labels = classifier.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Accuracy using Word2Vec Naive Bayes: {accuracy}")



if __name__ == "__main__":
    # main()
    NB_word2vec_4()
