from matplotlib import pyplot as plt
from time import time 
from sklearn.feature_extraction.text import CountVectorizer
from autocorrect import spell
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from pandas_datareader import data as data
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd 
import nltk
import re as re
import numpy as np
import json
import codecs
import os
import string
import collections
import Dataset

nltk.download('punkt')
nltk.download('stopwords')

os.system('cls')

def loadBookingData(filename = "booking.csv"):
    try:
        startTime = time()
        dataframe = pd.read_csv(filename,encoding='utf-8')
        dataframe = pd.DataFrame(dataframe)
        finishingTime = time()
        print("Data loaded correctly in {0:02f} sec".format(finishingTime-startTime))
        return dataframe
    except:
        print("Exception: Load data failed ")


def createDataset(dataframe):
    data = dataframe.iloc[:,1]
    label = dataframe.iloc[:,-1]
    print("Create Dataset => Length: {0} entries".format(len(data)))
    return data,label


def findStopWord(word):
    try: 
        stopwords = json.load(codecs.open('stopwords.json', 'r', 'utf-8-sig'))
    except:
        print("Loading stopwords.json failed")
    
    for lang in stopwords:
        if(word in stopwords[lang]):
            return True
        else:
            return False

def createListStopWords():
    stopwords = json.load(codecs.open('stopwords.json', 'r', 'utf-8-sig'))
    listStopwords = list() 
    for keys in stopwords.keys():
        for word in stopwords[keys]:
            listStopwords.append(word)
    listStopwords = np.array(listStopwords)
    listStopwords = np.unique(listStopwords)
    np.save("listStopWords",listStopwords)

def loadListStopWords():
    try:
        stopwords = np.load("listStopWords.npy")
        return stopwords
    except: 
        print("Error: list of stop words not loaded")


def preprocessingData(data):

    #allow only alphabetic characters
    data = re.sub('[^A-Za-z]', ' ', data)

    #convert to lowercase all characters
    data = data.lower()

    #remove stop words
    tokenized_data = word_tokenize(data) #make a list of words 
    for word in tokenized_data: 
        if(findStopWord(word)):
            tokenized_data.remove(word)

    #stemming: remove variations of the same word
    #stemmer = PorterStemmer()
    #for i in range(len(tokenized_data)):
        #tokenized_data[i] = stemmer.stem(spell(tokenized_data[i]))

    data = " ".join(tokenized_data)
    return data


def preprocessingData2(data):
    #convert to lowercase all characters 
    data = data.lower()

    #tokenize
    tokens = nltk.word_tokenize(data)

    tokens_words = [w for w in tokens if w.isalpha()]

    #stemming: remove variations of the same word
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in tokens_words]

    #remove stop words
    #meaningful_words = [w for w in stemmed_words if(findStopWord(w)==False)]

    #data_cleaned = (" ".join(meaningful_words))
    data_cleaned = (" ".join(stemmed_words))

    return data_cleaned

def preprocessingData3(X_data):
    X_dataCleaned = list()
    for data in X_data:
        tokens = nltk.word_tokenize(data)
        tokens_words = [w for w in tokens if w.isalpha()]
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in tokens_words]
        data_cleaned = (" ".join(stemmed_words))
        X_dataCleaned.append(data_cleaned)
    return X_dataCleaned

def processingDataset(X_data, y_label, batchSize = 50000):
    print('\n')
    #preprocessing data 
    listData = list()

    if(batchSize < 500):
        print("ERROR: batch size must be greater than 5000")
    numBatch = len(X_data) // batchSize
    print("[STEP 1] - Preprocessing data")
    i = 1
    startIdx = 0
    endIdx = batchSize-1
    for batch in range(numBatch):
        print("\n\n========= BATCH N. %d =========="%batch)
        startPreprocessing = time()
        dataBatch = X_data[startIdx:endIdx]
        print("\n[STEP 1] - Preprocessing data")
        for data in dataBatch:
            if (i % 1000 == 0):
                print("Pre-processing of %d/%d" %(i,len(X_data)))
            #print("BEFORE: %s" % data)
            dataProcessed = preprocessingData2(data)
            #print("AFTER: %s" % dataProcessed)
            #print('\n')
            listData.append(dataProcessed)
            i = i + 1
        finishPreprocessing = time()
        print("Preprocessing data - Completed. Total Time: {0:02f} sec".format(finishPreprocessing-startPreprocessing))
        print('\n')
        #extracting features
        print("[STEP 2] - Extracting feature")
        startTime = time() 
        vectorizer = CountVectorizer() 
        X = vectorizer.fit(listData)
        finishingTime = time()
        print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
        print('\n')
        #split training set and test set 
        print("[STEP 3] - Split data in training set and test set")
        X_Train, y_Train, X_Test, y_Test = splitDataset(X,y_label)
        print('\n')
        #save features
        print("[STEP 4] - Save features...")
        saveFeatures(X_Train, y_Train, X_Test, y_Test, batch)
        startIdx = startIdx+batchSize
        endIdx = endIdx+batchSize
        print('\n')
    print("All done.")

    return X 

def processingDatasetTry(X_data, y_label):
    print('\n')
    #preprocessing data 
    listData = list()
    i = 1
    startPreprocessing = time()

    print("\n[STEP 1] - Preprocessing data")
    for data in X_data:
        if (i % 1000 == 0):
            print("Pre-processing of %d/%d" %(i,len(X_data)))
        #print("BEFORE: %s" % data)
        dataProcessed = preprocessingData2(data)
        #print("AFTER: %s" % dataProcessed)
        #print('\n')
        listData.append(dataProcessed)
        i = i + 1
    finishPreprocessing = time()
    print("Preprocessing data - Completed. Total Time: {0:02f} sec".format(finishPreprocessing-startPreprocessing))
    print('\n')
    #extracting features
    print("[STEP 2] - Extracting feature")
    startTime = time() 
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(listData).toarray()

    finishingTime = time()
    print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
    print('\n')
    #split training set and test set 
    print("[STEP 3] - Split data in training set and test set")
    X_Train, y_Train, X_Test, y_Test = splitDataset(X,y_label)
    print('\n')
    #save features
    print("[STEP 4] - Save features...")
    saveFeatures(X_Train, y_Train, X_Test, y_Test)
    print('\n')
    print("All done.")

    return X 

def processingDatasetTry2(X_data, y_label):
    print('\n')
    #preprocessing data 
    startPreprocessing = time()
    #extracting features
    print("[STEP 1] - Extracting feature")
    startTime = time() 
    stopwords = loadListStopWords()
    X_dataList = X_data.tolist()
    vectorizer = CountVectorizer(stop_words=stopwords, preprocessor=preprocessingData3)
    X = vectorizer.fit_transform(X_dataList).toarray()
    finishingTime = time()
    print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
    print('\n')
    #split training set and test set 
    print("[STEP 2] - Split data in training set and test set")
    X_Train, y_Train, X_Test, y_Test = splitDataset(X,y_label)
    print('\n')
    #save features
    print("[STEP 3] - Save features...")
    saveFeatures(X_Train, y_Train, X_Test, y_Test)
    print('\n')
    print("All done.")

    return X 

def processingDatasetTFIDF(X_data, y_label, batchSize = 20000):
    print('\n')
    #preprocessing data 
    listData = list()
    if(batchSize < 5):
        print("ERROR: batch size must be greater than 5000")
    numBatch = len(X_data) // batchSize
    i = 1
    startIdx = 0
    endIdx = batchSize-1
    for batch in range(numBatch):
        print("\n\n========= BATCH N. %d =========="%batch)
        startPreprocessing = time()
        dataBatch = X_data[startIdx:endIdx]
        print("\n[STEP 1] - Preprocessing data")
        for data in dataBatch:
            if (i % 100 == 0):
                print("Pre-processing of %d/%d" %(i,len(X_data)))
            #print("BEFORE: %s" % data)
            dataProcessed = preprocessingData(data)
            #print("AFTER: %s" % dataProcessed)
            #print('\n')
            listData.append(dataProcessed)
            i = i + 1
        finishPreprocessing = time()
        print("Preprocessing data - Completed. Total Time: {0:02f} sec".format(finishPreprocessing-startPreprocessing))
        print('\n')
        #extracting features
        print("[STEP 2] - Extracting feature")
        startTime = time() 
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(listData).toarray()
        finishingTime = time()

        print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
        print('\n')
    
        km_model = KMeans(n_clusters=37)
        km_model.fit(X)
 
        clustering = collections.defaultdict(list)
 
        for idx, label in enumerate(km_model.labels_):
            clustering[label].append(idx)
 
        #split training set and test set 
        print("[STEP 3] - Split data in training set and test set")
        X_Train, y_Train, X_Test, y_Test = splitDataset(X,y_label)
        print('\n')
        #save features
        print("[STEP 4] - Save features...")
        saveFeatures(X_Train, y_Train, X_Test, y_Test, batch)
        startIdx = startIdx+batchSize
        endIdx = endIdx+batchSize
        print('\n')

    print("All done.")

    return X, clustering 

def processingDatasetTFIDFTry(X_data, y_label):
    print('\n')
    #preprocessing data 
    listData = list()
    i = 1
    startPreprocessing = time()

    print("\n[STEP 1] - Preprocessing data")
    for data in X_data:
        if (i % 1000 == 0):
            print("Pre-processing of %d/%d" %(i,len(X_data)))
        #print("BEFORE: %s" % data)
        dataProcessed = preprocessingData2(data)
        #print("AFTER: %s" % dataProcessed)
        #print('\n')
        listData.append(dataProcessed)
        i = i + 1
    finishPreprocessing = time()
    print("Preprocessing data - Completed. Total Time: {0:02f} sec".format(finishPreprocessing-startPreprocessing))
    print('\n')
    #extracting features
    print("[STEP 2] - Extracting feature")
    startTime = time() 
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(listData).toarray()
    finishingTime = time()

    print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
    print('\n')
   
    km_model = KMeans(n_clusters=37)
    km_model.fit(X)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    #split training set and test set 
    print("[STEP 3] - Split data in training set and test set")
    X_Train, y_Train, X_Test, y_Test = splitDataset(X,y_label)
    print('\n')
    #save features
    print("[STEP 4] - Save features...")
    saveFeatures(X_Train, y_Train, X_Test, y_Test)
    print('\n')

    print("All done.")

    return X, clustering 

def splitDataset(data, label, percentage_split = 0.7): 
    index_of_splitting = int(len(data) * 0.7)
    data = np.array(data)
    label = np.array(label)
    X_Train = data[:index_of_splitting]
    y_Train = label[:index_of_splitting]
    X_Test = data[index_of_splitting:]
    y_Test = label[index_of_splitting:]

    print("Length Training Set: {0} ({2} percent); Length Test Set {1} ({3} percent)".format(X_Train.shape,X_Test.shape, int(percentage_split*100), int((1-percentage_split)*100)))
    return X_Train, y_Train, X_Test, y_Test


def saveFeatures(X_Tr,y_Tr,X_Ts,y_Ts):
    try:
        nameFile = "X_Training"
        np.save(nameFile,X_Tr)
        nameFile = "X_Test"
        np.save(nameFile,X_Ts)
        nameFile = "y_Training"
        np.save(nameFile,y_Tr)
        nameFile = "y_Test"
        np.save(nameFile,y_Ts)
    except:
        print("Error: Features not saved")


#1) load dataframe
dataframe = loadBookingData()
#2) make dataset (data + label) 
X_data,y_label = createDataset(dataframe)
#3) processing dataset 
#_,clustering = processingDatasetTFIDF(X_data,y_label)
processingDatasetTry(X_data[:25000],y_label[:25000])
