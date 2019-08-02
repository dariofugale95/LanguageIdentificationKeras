from Enviroment import Enviroment as env
import pickle

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