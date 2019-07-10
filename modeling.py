from Enviroment import Enviroment as env

from keras.layers import Input, LSTM, Embedding, Dense, Dropout
from keras.models import load_model, Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

class LanguageClassifier():
    def __init__(self,x_train,y_train,x_test,y_test, vocab_size = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.vocab_size = vocab_size
        
        self.batch_size = env().BATCH_SIZE
        self.embed_dim = env().EMBEDDING_DIM 
        self.epochs = env().EPOCHS
        self.optimizer = env().OPTIMIZER
        self.units = env().UNITS
        self.model_filename = env().MODEL_FILENAME

        self.model = Sequential()

    def _build_baseline(self):
        # self.model.add(Embedding(self.vocab_size, env().EMBEDDING_DIM, input_length = self.x_train.shape[1]))
        # self.model.add(LSTM(256, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))
        # self.model.add(LSTM(256, dropout = 0.1, recurrent_dropout = 0.1))
        # self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

        self.model.add(Dense(500, input_dim = self.x_train.shape[1], activation="sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(300, activation="sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(100, activation="sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.y_train.shape[1], activation="softmax", name = "OutputLayer"))

        # input_layer = Input(shape=(self.x_train.shape[1],), name = "InputLayer")
        # embedding_layer = Embedding(input_dim = self.x_train.shape[1], output_dim = self.embed_dim, name = "EmbeddingLayer")
        # lstm_layer = LSTM(units = self.units, return_sequences = False, return_state = True, name = "LSTMLayer")
        # lstm_outputs, hidden_state, cell_state = lstm_layer(embedding_layer(input_layer))
        # encoder_states = [hidden_state, cell_state]
        # dense_layer_1 = Dense(units=self.units, activation = "relu", name="Dense1")
        # dense_layer_2 = Dense(units=38, activation = "softmax", name="Dense2")

        # lstm_outputs = dense_layer_2(Dropout(rate=.1)(dense_layer_1(Dropout(rate=.1)(lstm_outputs))))
        # self.model = Model(input_layer, lstm_outputs)

    def _compile_model(self):
        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self._build_baseline()
        self.model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
            )

    def _save_checkpoint(self):
        filepath = env().MODEL_FILENAME+""+"checkpoint.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        return callbacks_list

    def train_model(self):
        self._compile_model()
        print(self.model.summary())
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test,self.y_test),
            callbacks=self._save_checkpoint()
            )

    def save_model(self):
        if(self.reverse):
            model_json = self.model.to_json()
            with open(self.model_filename+""+".json", "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights(self.model_filename+""+".h5")

            