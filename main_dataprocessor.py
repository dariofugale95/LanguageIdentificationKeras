from dataprocessing import DataReader, DataGenerator
from Enviroment import Enviroment as env
from modeling import LanguageClassifier
from keras.models import load_model
import pickle 
import numpy as np

def load_vocabularies():
    """ Load vocabularies from file

    :return: char_to_indx, idx_to_char, maq_seq_len
    """
    with open(env().path_to_vocabularies, "rb") as f_pickle:
        char_to_idx = pickle.load(f_pickle)
        idx_to_char = pickle.load(f_pickle)
        max_seq_len = pickle.load(f_pickle)
    return char_to_idx, idx_to_char, max_seq_len

def load_data(): 
    """ Load the training and test arrays from file

    :return: X_training, X_test, y_training, y_test
    """
    with open(env().path_to_arrays, "rb") as f_pickle:
        X_tr = pickle.load(f_pickle)
        X_te = pickle.load(f_pickle)
        y_tr = pickle.load(f_pickle)
        y_te = pickle.load(f_pickle)
    return X_tr, X_te, y_tr, y_te
    
def main():
    # dr = DataReader(env().data_dir)
    # data, labels = dr.create_dataset()
    # dr.save_data_csv()
    # data, labels = dr.load_dataset()
    # dg = DataGenerator(data, labels)
    # data, labels = dg.generate_data()
    # X_train, X_test, y_train, y_test = dg.split_train_test(env().DIM_TEST)
    # dg.save_data(X_train, X_test, y_train, y_test)
    # dg.save_vocabularies()
    X_train, X_test, y_train, y_test = load_data()
    char_to_idx, idx_to_char, max_seq_len = load_vocabularies()

    X_train = X_train[:,:50]
    X_test = X_test[:,:50]

    print(len(char_to_idx))
    print(X_train.shape)
    langclassifier = LanguageClassifier(X_train,y_train,X_test,y_test, len(char_to_idx))
    langclassifier.train_model()
    langclassifier.save_model()
    # langclassifier = load_model("lang_classifiercheckpoint.h5")
    # predictions  = langclassifier.predict(X_test)


if __name__ == '__main__':
    main()