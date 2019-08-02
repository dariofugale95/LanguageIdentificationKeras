from matplotlib import pyplot as plt
from time import time 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KDTree
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.feature_extraction.text import CountVectorizer
import re as re
import numpy as np
import json
import os 
import seaborn as sns; sns.set()
from autocorrect import spell
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

os.system('cls')

def loadBoW(batch):
    X_Training = np.load("X_Training"+str(batch)+".npy")
    X_Test = np.load("X_Test"+str(batch)+".npy")
    y_Training = np.load("y_Training"+str(batch)+".npy")
    y_Test = np.load("y_Test"+str(batch)+".npy")
    np.array(X_Training)
    np.array(X_Test)
    np.array(y_Training)
    np.array(y_Test)
    return X_Training, y_Training, X_Test, y_Test

def loadBoWTry():
    print("Loading BOW...\n")
    X_Training = np.load("X_Training.npy")
    X_Test = np.load("X_Test.npy")
    y_Training = np.load("y_Training.npy")
    y_Test = np.load("y_Test.npy")
    print("BOW loaded!\n")
    np.array(X_Training)
    np.array(X_Test)
    np.array(y_Training)
    np.array(y_Test)
    return X_Training, y_Training, X_Test, y_Test



def plotBowRepresentation(X):
    plt.plot(X)
    plt.show()

def plotConfusionMatrix(y_Test, y_Predicted):
    confusion_matrix = confusion_matrix(y_Test,y_Predicted)
    print("Accuracy: %f" %accuracy)
    sns.heatmap(confusion_matrix.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show()

# IT = italiano (0), EN = inglese (1), DE = tedesco (2), ES = spagnolo (3), FR = francese (4), PT = portoghese (5), NL = olandese (6)
# PL = polacco (7), RU = russo (8), HE = ebraico (9), NO = norvegese (10), CS = ceco (11), SL = sloveno (12), CA = catalano (13), RO = rumeno (14)
# ET = estone (15), KO = coreano (16), SV = svedese (17), DA = danese (18), ZH = cinese (19), TR = turco (20), JA = giapponese (21), HU = ungherese (22)
# HR = croato (23), LT = lituano (24), LV = lettone (25), FI = finlandese (26), UK = ucraino (27), SR = serbo (28), BG = bulgaro (29), SK = slovacco (30)
# EL = greco (31), AR = arabo (32), IS = islandese (33), MS = malese (34), TH = thai (35), IN = indonesiano (36), VI = vietnamita (37)

def getLanguage(label):
    langDict = {0 : 'italiano', 1 : 'inglese', 2 : 'tedesco', 3 : 'spagnolo', 4 : 'francese', 5 : 'portoghese', 6 : 'olandese',
            7 : 'polacco', 8 : 'russo', 9 : 'ebraico', 10 : 'norvegese', 11 : 'ceco', 12 : 'sloveno', 13 : 'catalano', 14 : 'romeno',
            15 : 'estone', 16 : 'coreano', 17 : 'svedese', 18 : 'danese', 19 : 'cinese', 20 : 'turco', 21 : 'giapponese', 22 : 'ungherese',
            23 : 'croato', 24 : 'lituano', 25 : 'lettone', 26 : 'finlandese', 27 : 'ucraino', 28 : 'serbo', 29 : 'bulgaro', 30 : 'slovacco',
            31 : 'greco', 32 : 'arabo', 33 : 'islandese', 34 : 'malese', 35 : 'tailandese', 36 : 'indonesiano', 37 : 'vietnamita'}
    return langDict[label]

def predictLanguage(sentence, model):
    X_sentence = processingSentence(sentence)
    print(np.shape(X_sentence))
    y_Predicted = model.predict(X_sentence)
    return y_Predicted[0]

def processingSentence(sentence):
    print('\n')
    #preprocessing data 
    i = 1
    print("\n[STEP 1] - Preprocessing data")
    startPreprocessing = time()
    sentenceProcessed = preprocessingData2(sentence)
    finishPreprocessing = time()
    print("Preprocessing data - Completed. Total Time: {0:02f} sec".format(finishPreprocessing-startPreprocessing))
    print('\n')
    #extracting features
    print("[STEP 2] - Extracting feature")
    startTime = time() 
    vectorizer = CountVectorizer() 
    sentenceProcessed = [sentenceProcessed]
    X_sentence = vectorizer.fit_transform(sentenceProcessed)
    finishingTime = time()
    print("Extracting features - Completed. Total Time: {0:0.2f} sec".format(finishingTime-startTime))
    print('\n')
    print("All done.")

    return X_sentence

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


X_Training, y_Training, X_Test, y_Test = loadBoWTry()

classifier = NB(alpha = 0.03)
print("Fitting NB model\n")
classifier.fit(X_Training, y_Training)

# Predict Class
print("Predict class\n")
y_Predicted = classifier.predict(X_Test)

# Accuracy 
print(np.shape(X_Test),np.shape(y_Predicted))
accuracy = accuracy_score(y_Test,y_Predicted)

print("Accurcay %f" %accuracy)

np.save("naivebayesclassifier",classifier)
#sentence = "ciao sono dario e ho ventiquattro anni, posto molto bello da vedere"
#y_sentence_predicted = predictLanguage(sentence, classifier)
#print(y_sentence_predicted)

