from dataprocessing import DataReader, DataGenerator
from textprocessing import TextPreprocessor as TP
from Enviroment import Enviroment as env
from sklearn import preprocessing
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import pickle
import os 
import numpy as np

def load_data(): 
    with open(env().path_to_arrays, "rb") as f_pickle:
        X_tr = pickle.load(f_pickle)
        X_te = pickle.load(f_pickle)
        y_tr = pickle.load(f_pickle)
        y_te = pickle.load(f_pickle)
        word_to_idx = pickle.load(f_pickle)
        idx_to_word = pickle.load(f_pickle)
        max_words = pickle.load(f_pickle)
    return X_tr, X_te, y_tr, y_te, word_to_idx, idx_to_word, max_words

def predict_lang(sentence, word_to_idx, idx_to_word, max_seq_len):
    # Clean the sentence
    tp = TP(sentence)
    sentence_cleaned = tp.preprocess_text()

    # inizialize input to model
    X_sentence = np.zeros((1,max_seq_len), dtype='float32') 

    for word_idx, word in enumerate(sentence_cleaned.split()):
        X_sentence[0, word_idx] = word_to_idx[word]

    print(X_sentence)
    # load model classifier
    langclassifier = load_model("lang_classifiercheckpoint.h5")
    predictions = langclassifier.predict(X_sentence)
    print(predictions)
    # Get the highest prediction
    y_pred = np.argmax(predictions,axis=1)
    
    return y_pred

def main():
    _, _, _, _, word_to_idx, idx_to_word, max_words_length = load_data()
    y_pred = predict_lang("stai sereno sono molto stanco devo andare a lavorare", word_to_idx, idx_to_word, max_words_length)
if __name__ == '__main__':
    main()