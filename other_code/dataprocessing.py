from Enviroment import Enviroment as env
from textprocessing import TextPreprocessor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import pickle
import numpy as np
import random
import os 

class DataReader(object):
    """description of class"""
    def __init__(self, data_directory_path):
        self.data_directory_path = data_directory_path
        self.list_dataset_files = sorted(os.listdir(data_directory_path))
        
        self.max_list_len = 0 
        self.data = []
        self.labels = []

    def _read_data_files(self):
        list_all_data = []
        for data_file in self.list_dataset_files:
            try:
                with open(self.data_directory_path+data_file, mode="rt", encoding="utf-8") as f:
                    list_all_data.append(f.read().split('\n'))
                    f.close()
            except Exception as ex:
                print("Error: ", ex)
        return list_all_data

    def _split_data(self):
        list_all_data = self._read_data_files()
        english_texts, langx_texts, list_labels = [],[],[]
        label = 1
        for list_pairs in list_all_data:
            self._update_max_list_len(len(list_pairs)) 
            # last item is an empty row so the split() function return with error 
            for pair in list_pairs[:-1]:
                # split by tab char '\t'
                pair_splitted = pair.split(sep='\t')
                # preprocessing 
                tp = TextPreprocessor(pair_splitted[0])
                english_row = tp.preprocess_text()
                tp = TextPreprocessor(pair_splitted[1])
                langx_row = tp.preprocess_text()
                english_texts.append(english_row)
                langx_texts.append(langx_row)
                list_labels.append(label)
            label = label + 1 

        return english_texts, langx_texts, list_labels

    def _update_max_list_len(self, len_list):
        if len_list > self.max_list_len: 
            self.max_list_len = len_list
    
    def _zerolistmaker(self, n):
        listofzeros = [0] * n
        return listofzeros

    def _shuffle_data(self, _data, _labels):
        zipped_list = list(zip(_data, _labels))
        random.shuffle(zipped_list)
        _data, _labels = zip(*zipped_list)
        return _data, _labels

    def create_dataset(self, shuffle = True):
        english_texts, langx_texts, list_labels = self._split_data()
        # select "max_list_len" random items from english_texts  
        random.shuffle(english_texts)
        english_texts = english_texts[:self.max_list_len]
        # concat english texts with all texts of other languages
        _data = english_texts + langx_texts 
        # concat english texts labels with list_labels
        _labels = self._zerolistmaker(self.max_list_len) + list_labels
        
        if(shuffle):
            _data, _labels = self._shuffle_data(_data, _labels)


        self.data = _data
        self.labels= _labels
        return self.data, self.labels

    def save_data_csv(self):
        if(len(self.data) > 0):
            try:
                with open(env().dataset_csv_path, mode = "w", encoding="utf-8") as f_csv: 
                    for i in range(len(self.data)):
                        f_csv.write("{}\t{}\n".format(self.data[i], self.labels[i]))
                    f_csv.close()
            except Exception as ex:
                print("Error: dataset not save - ", ex)
        else: 
            print("Empty dataset!\n")

    def load_dataset(self):
        data_list, labels_list = [],[]
        with open(env().dataset_csv_path, mode="rt", encoding="utf-8") as f_csv:
            lines = f_csv.read().split('\n')
            for line in lines[:-1]:
                x, y = line.split('\t') 
                data_list.append(x)
                labels_list.append(y)
            f_csv.close()
        # restore fields of object datareader
        self.data = data_list
        self.label = labels_list
        return self.data, self.label

class DataGenerator():
    def __init__(self, data, labels): 
        self.data = data
        self.labels = labels
        self.max_seq_len = 0 
        
        self.word_to_index = {} # {"Hello":1, "World":2,....}
        self.index_to_word = {} # {1:"Hello", 2:"World",....}
    
        self.X_data = []
        self.y_data = []

    def _update_max_seq_len(self, len_words_seq):
        if(len_words_seq > self.max_seq_len):
            self.max_seq_len = len_words_seq

    def get_max_seq_len(self):
        return self.max_seq_len

    def _create_word_vocabulary(self):
        for text in self.data:
            words = text.split(sep=" ")
            self._update_max_seq_len(len(words))
            for word in words:
                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)+1
                    self.index_to_word[len(self.word_to_index)+1] = word 
                    
    def generate_data(self):
        self._create_word_vocabulary()

        #initialize X_data 
        self.X_data = np.zeros((len(self.data), self.get_max_seq_len()), dtype='float32') 

        for n_sample_text, text in enumerate(self.data):
            for word_idx, word in enumerate(text.split()):
                self.X_data[n_sample_text, word_idx] = self.word_to_index[word]

        # normalize data
        standard_scaler = preprocessing.StandardScaler().fit(self.X_data)
        self.X_data = standard_scaler.transform(self.X_data)   
        
        # one-hot targets 
        self.y_data = to_categorical(self.labels, num_classes=None)

        return self.X_data, self.y_data
    
    def split_train_test(self, percentage_validation, shuffle = True): 
        X_train, X_test, y_train, y_test = train_test_split(self.X_data, 
                                                            self.y_data, 
                                                            test_size = percentage_validation, 
                                                            shuffle = shuffle)
        return X_train, X_test, y_train, y_test

    def save_data(self, X_tr, X_te, y_tr, y_te):
        with open(env().path_to_arrays, 'wb') as f_pickle:
            pickle.dump(X_tr, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(X_te, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_tr, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_te, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.word_to_index, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.index_to_word, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.max_seq_len, f_pickle, protocol = pickle.HIGHEST_PROTOCOL)



    

        

