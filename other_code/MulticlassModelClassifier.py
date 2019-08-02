###################################
#            Library              #
###################################
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import SGD, Nadam, Adam
from time import time 
from Dataset import getBookingDataset, saveBookingData, loadBookingData
import pandas as pd 
import numpy as np

###################################
#         Input Variabiles        #
###################################
#Network params
size_test_set = 0.1 #90% training set, 10% test set
num_epochs = 10
neurons = [100]

#Credential MongoClient
username = 'amodto'
password = 'pwamod2018'


def preprocessingData(text_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    tokens = tokenizer.texts_to_sequences(text_data)

    # calculate row with "max words" in dataframe
    max_words = max(len(tokens[i][:]) for i in range(len(tokens[:])))
    
    # Add 0 pad
    zero_array = np.zeros((len(tokens),max_words), dtype = int)
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if(tokens[i][j] > 0):
                zero_array[i][j] = zero_array[i][j] + tokens[i][j] 
    tokens = zero_array
    tokens = np.array(tokens).T.tolist()
    return tokens, max_words


def makeInputData(): 
    dataframe = loadBookingData("booking_tokenized2.csv")
    X = dataframe.iloc[:,1:50]
    y = dataframe.loc[:,"LangLabel"]
    
    #print(X.head(3))
   
   # split dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size_test_set,shuffle=True)
    y_train = y_train.values
    y_test = y_test.values
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(X_train.shape)
    return X_train, y_train, X_test, y_test


def build_LSTM_Model_Classifier(X_train, y_train, X_test, y_test):
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    classifier = Sequential()
    classifier.add(LSTM(76, input_shape=(1,X_test.shape[2])))
    classifier.add(Dense(38, activation="softmax"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    classifier.compile(optimizer="nadam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(X_train,y_train, epochs = 100, validation_data=(X_test,y_test), shuffle=True, batch_size = 512)

    #classifier = Sequential()
    #classifier.add(Dense(5000, activation='relu', input_dim = 1, output_dim = 1))
    #classifier.add(Dropout(0.1))
    #classifier.add(Dense(600, activation='relu', input_dim = 1, output_dim = 38))
    #classifier.add(Dropout(0.1))
    #classifier.add(Dense(38, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    #classifier.compile(loss='categorical_crossentropy', optimizer=sgd)

    #classifier.fit(X_train[:], y_train[:], epochs=5)

    #score, accuracy = classifier.evaluate(X_train, y_train)
    #print('Train Score: ', score)
    #print('Train Accuracy: ', accuracy)
    #score, accuracy = classifier.evaluate(X_test, y_test)
    #print('Test Score: ', score)
    #print('Test Accuracy: ', accuracy)



    #y_pred_labels = classifier.predict_classes(X_test[:])
    
    return classifier
    
def build_Model_Classifier(X_train, y_train, X_test, y_test):
    
    classifier = Sequential()
    classifier.add(Dense(38, activation='relu', output_dim = 38))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(38, activation='softmax', input_dim = 38))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    classifier.compile(loss='categorical_crossentropy', optimizer=sgd)

    classifier.fit(X_train, y_train, epochs=5)

    score, accuracy = classifier.evaluate(X_train, y_train)
    print('Train Score: ', score)
    print('Train Accuracy: ', accuracy)
    score, accuracy = classifier.evaluate(X_test, y_test)
    print('Test Score: ', score)
    print('Test Accuracy: ', accuracy)

    y_pred_labels = classifier.predict_classes(X_test)
    
    return classifier, y_pred_labels


def main(): 
    #dataframe = getBookingDataset(username,password)
    #data = dataframe.loc[:,"Raw"]
    #labels = dataframe.loc[:,"LangLabel"]
    #data_tokenized, max_words = preprocessingData(data)
    #new_dataframe = pd.DataFrame()
    #for i in range(max_words):
    #    new_dataframe[str(i)] = data_tokenized[:][i]
    #new_dataframe["LangLabel"] = labels
    #del(data)
    #del(data_tokenized)
    #del(dataframe)
    #saveBookingData(new_dataframe,"booking_tokenized2.csv")

    dataframe = loadBookingData("booking_tokenized.csv")
    #X = dataframe.loc[:,"Data"]
    #y = dataframe.loc[:,"LangLabel"]

    X_train,y_train,X_test,y_test = makeInputData()

    #print(X_train[0][:].shape) #374 ndarray

    classifier = build_LSTM_Model_Classifier(X_train,y_train,X_test,y_test)
    # serialize model to JSON
    model_json = classifier.to_json()
    with open("classifier.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("classifier.h5")
    print("Saved model to disk")
 
    

if __name__=="__main__":
    main()
