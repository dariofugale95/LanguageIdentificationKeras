import os

class Enviroment():
    def __init__(self):
        self.working_dir = os.getcwd()
        self.data_dir = self.working_dir+"/data/"
        self.saved_items_dir = self.working_dir+"/saved_items/"
        self.dataset_csv_path = self.working_dir+"/dataset.csv"
        self.path_to_arrays = self.saved_items_dir+""+"saved_items.pkl"

        self.DIM_TEST = 0.1 #90 % training set, 10% test(validation) set 

        # Hyperparameters 
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.EMBEDDING_DIM = 200
        self.UNITS = 1024

        self.OPTIMIZER = "adam"
        self.MODEL_FILENAME = "lang_classifier"

