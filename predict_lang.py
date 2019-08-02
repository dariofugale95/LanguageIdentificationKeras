from dataprocessing import DataReader, DataGenerator
from textprocessing import TextPreprocessor as TP
from Enviroment import Enviroment as env
from keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf
import logging
import onnxmltools
import onnx

tf.get_logger().setLevel(logging.ERROR)

def load_vocabularies():
    """ Load vocabularies from file

    :return char_to_indx, idx_to_char, maq_seq_len
    """
    with open(env().path_to_vocabularies, "rb") as f_pickle:
        char_to_idx = pickle.load(f_pickle)
        idx_to_char = pickle.load(f_pickle)
        max_seq_len = pickle.load(f_pickle)
    return char_to_idx, idx_to_char, max_seq_len

def _get_language_string(index):
    """ Retrieve the language by index

    :param: index (dictionary key)

    :return: a string containing the name of the language
    """

    index_to_lang = {0:"English", 1:"Arabic",2:"Bulgarian",3:"Catalan",4:"Czech",5:"Chinese (Mandarin)",6:"Danish ",7:"German",8:"Greek",9:"Estonian",
                    10:"Finnish",11:"French",12:"Hebrew",13:"Croatian",14:"Hungarian",15:"Indonesian",16:"Icelandic",17:"Italian",18:"Japanese",19:"Korean",
                    20:"Lithuanian",21:"Latvian",22:"Dutch",23:"Norwegian (Bokmål)",24:"Polish",25:"Portuguese",26:"Romanian",27:"Russian",28:"Slovak",29:"Slovenian",
                    30:"Spanish",31:"Serbian",32:"Swedish",33:"Thai",34:"Turkish",35:"Ukrainian",36:"Vietnamese",37:"Malay"}
    return index_to_lang[index]

def predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len):
    """ This function predicts the language given a sentence as input.
    
    STEP 1: preprocessing of the sentence;
    
    STEP 2: transformation of the sentence into input for the prediction model;
    
    STEP 3: language predictions;

    STEP 4: calculate the language with the highest probability;

    :param: sentence (string)
    :param: char_to_idx (dict) 
    :param: idx_to_char (dict) 
    :param: max_seq_len (int) 

    :return: the predicted language
    """

    # read only first 50 characters
    sentence = sentence[:50]

    # Clean the sentence
    tp = TP(sentence)
    sentence_cleaned = tp.preprocess_text()

    sentence_cleaned = sentence_cleaned.replace('"',"'")

    # check unknown char in sentence
    for char in sentence_cleaned:
        if char not in set(char_to_idx.keys()):
            print(char)
            sentence_cleaned = sentence_cleaned.replace(char,"")
    
    # inizialize input to model
    X_sentence = np.zeros((1,50), dtype='float32') 
    for char_idx, char in enumerate(sentence_cleaned):
        X_sentence[0, char_idx] = char_to_idx[char]

    # load model classifier
    langclassifier = load_model("lang_classifierLARGE.h5")
    predictions = langclassifier.predict(X_sentence)
    # Get the highest prediction
    y_pred = np.argmax(predictions)
    string_lang = _get_language_string(y_pred)
    return string_lang
    
def main():

    # ======== CONVERT TO ONNX ============
    # langclassifier = load_model("lang_classifierLARGE.h5")
    # langclassifier_onnx = onnx.convert_keras(langclassifier)
    # onnx.save_model(langclassifier_onnx, "langclassifier.onnx")


    char_to_idx, idx_to_char, max_seq_len = load_vocabularies()

    # ============== TEST ================== 

    # English
    sentence = "This hotel is very spacious. The service is excellent but the price is too high."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Arabic
    sentence = "هذا الفندق واسع للغاية. الخدمة ممتازة ولكن السعر مرتفع للغاية."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Bulgarian
    sentence = "Този хотел е много просторен. Услугата е отлична, но цената е твърде висока."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Catalan
    sentence = "Aquest hotel és molt espaiós. El servei és excel·lent, però el preu és massa alt."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Czech
    sentence = "Tento hotel je velmi prostorný. Služba je vynikající, ale cena je příliš vysoká."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    Chinese (Mandarin)
    sentence = "这家酒店非常宽敞"
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Danish
    sentence = "Dette hotel er meget rummeligt. Tjenesten er fremragende, men prisen er for høj."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # German
    sentence = "Das Hotel ist sehr geräumig. Der Service ist ausgezeichnet, aber der Preis ist zu hoch."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Greek
    sentence = "Αυτό το ξενοδοχείο είναι πολύ ευρύχωρο. Η υπηρεσία είναι εξαιρετική, αλλά η τιμή είναι πολύ υψηλή."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Estonian
    sentence = "See hotell on väga avar. Teenus on suurepärane, kuid hind on liiga kõrge."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Finnish
    sentence = "Tämä hotelli on erittäin tilava. Palvelu on erinomainen, mutta hinta on liian korkea."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # French
    sentence = "Cet hôtel est très spacieux. Le service est excellent mais le prix est trop élevé."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Hebrew
    sentence = "מלון זה הינו מרווח מאוד. השירות מצוין אבל המחיר גבוה מדי."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Croatian
    sentence = "Ovaj hotel je vrlo prostran. Usluga je izvrsna, ali cijena je previsoka."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Hungarian
    sentence = "Ez a szálloda nagyon tágas. A szolgáltatás kiváló, de az ár túl magas."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Indonesian
    sentence = "Hotel ini sangat luas. Layanan ini sangat baik tetapi harganya terlalu tinggi."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Icelandic
    sentence = "Þetta hótel er mjög rúmgott. Þjónustan er góð en verðið er of hátt."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Italian
    sentence = "Questo albergo è molto spazioso. Il servizio è eccellente ma il prezzo è troppo alto."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    Japanese
    sentence = "このホテルはとても広々としています。サービスは優れているが価格が高すぎる。"
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    Korean
    sentence = "이 호텔은 매우 넓습니다. 서비스는 훌륭하지만 가격이 너무 비쌉니다."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

    # Lithuanian
    sentence = "Šis viešbutis yra labai erdvus. Paslauga yra puiki, bet kaina per didelė."
    print(sentence)
    string_lang = predict_lang(sentence, char_to_idx, idx_to_char, max_seq_len)
    print("%s\n"%string_lang)

if __name__ == '__main__':
    main()
