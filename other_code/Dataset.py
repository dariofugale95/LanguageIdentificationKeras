from pymongo import MongoClient
from pandas_datareader import data as data
from time import time
import random
import datetime as dt
import numpy as np
import pandas as pd 
import string
import urllib
import json
import csv
import bson

#Credential MongoClient
username = 'amodto'
password = 'pwamod2018'

def getBookingDataset(username, password):
    startTime = time()
    
    username = urllib.parse.quote_plus(username)
    password = urllib.parse.quote_plus(password)

    client = MongoClient("mongodb://%s:%sto@192.168.40.244:27017" %(username,password))
    db = client.get_database("booking")
    collection = db.get_collection("raw")

    print("Getting collection '{0}' from DB '{1}' ...".format(collection.name,db.name))
    
    cursor = collection.find({},{'_id':0,'BookingReviews.UserReview.Pos':1,'BookingReviews.UserReview.Neg':1,'BookingReviews.UserReview.Lang':1})

    rawList =  list()
    langList = list()
    langLabelList = list()

    for document in cursor:
        for i in range(len(document['BookingReviews'])):
            if(str(document['BookingReviews'][i]['UserReview']['Pos']) != "None" and len(str(document['BookingReviews'][i]['UserReview']['Pos'])) > 20):

                raw = str(document['BookingReviews'][i]['UserReview']['Pos']) # devo aggiungere anche 'Neg'
                lang = document['BookingReviews'][i]['UserReview']['Lang']
                lang, langLabel = addLanguageAbbreviation(lang) # add abbreviation language for example "IT : italiano" and also return label for abbreviation (labelLang) for example "IT -> 0" 
                rawList.append(raw)
                langList.append(lang)
                langLabelList.append(langLabel)
    
    data = {'Raw':rawList,'Lang':langList, 'LangLabel':langLabelList}

    dataFrame = pd.DataFrame(data=data)

    finishingTime = time()

    print("Data loaded correctly in {0:02f} sec".format(finishingTime-startTime))
    #print(len(rawList))
    return dataFrame


def saveBookingData(dataframe, filename = "booking.csv"):
    try:
        startTime = time()
        dataframe.to_csv(filename,encoding='utf-8')
        finishingTime = time()
        print("Data saved in '{0}' correctly in {1:02f} sec".format(filename, finishingTime-startTime))
    except IOError:
        print("Exception: I/O error")

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

# IT = italiano (0), EN = inglese (1), DE = tedesco (2), ES = spagnolo (3), FR = francese (4), PT = portoghese (5), NL = olandese (6)
# PL = polacco (7), RU = russo (8), HE = ebraico (9), NO = norvegese (10), CS = ceco (11), SL = sloveno (12), CA = catalano (13), RO = rumeno (14)
# ET = estone (15), KO = coreano (16), SV = svedese (17), DA = danese (18), ZH = cinese (19), TR = turco (20), JA = giapponese (21), HU = ungherese (22)
# HR = croato (23), LT = lituano (24), LV = lettone (25), FI = finlandese (26), UK = ucraino (27), SR = serbo (28), BG = bulgaro (29), SK = slovacco (30)
# EL = greco (31), AR = arabo (32), IS = islandese (33), MS = malese (34), TH = thai (35), IN = indonesiano (36), VI = vietnamita (37)

def addLanguageAbbreviation(lang):
    langDict = {'IT' : 'italiano', 'EN' : 'inglese', 'DE' : 'tedesco', 'ES' : 'spagnolo', 'FR' : 'francese', 'PT' : 'portoghese', 'NL' : 'olandese',
            'PL' : 'polacco', 'RU' : 'russo', 'HE' : 'ebraico', 'NO' : 'norvegese', 'CS' : 'ceco', 'SL' : 'sloveno', 'CA' : 'catalano', 'RO' : 'romeno',
            'ET' : 'estone', 'KO' : 'coreano', 'SV' : 'svedese', 'DA' : 'danese', 'ZH' : 'cinese', 'TR' : 'turco', 'JA' : 'giapponese', 'HU' : 'ungherese',
            'HR' : 'croato', 'LT' : 'lituano', 'LV' : 'lettone', 'FI' : 'finlandese', 'UK' : 'ucraino', 'SR' : 'serbo', 'BG' : 'bulgaro', 'SK' : 'slovacco',
            'EL' : 'greco', 'AR' : 'arabo', 'IS' : 'islandese', 'MS' : 'malese', 'TH' : 'tailandese', 'IN' : 'indonesiano', 'VI' : 'vietnamita'}

    keySet = list(langDict.keys())

    for i in range(len(langDict)):
        if lang == langDict[keySet[i]]:
            langMod = lang.replace(lang, keySet[i]+":"+langDict[keySet[i]])
            return langMod, i

def main():
    dataframe = getBookingDataset(username,password)        
    saveBookingData(dataframe)

if __name__ == '__main__':
    main()



