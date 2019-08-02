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
        self.model.add(Embedding(self.vocab_size, env().EMBEDDING_DIM, input_length = self.x_train.shape[1]))
        self.model.add(LSTM(56, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))
        self.model.add(LSTM(56, dropout = 0.1, recurrent_dropout = 0.1))
        self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

    def _compile_model(self):
        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self._build_baseline()
        self.model.compile(
            optimizer=adam_optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

    def _save_checkpoint(self):
        filepath = env().MODEL_FILENAME+""+"LARGE.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        return callbacks_list

    def train_model(self):
        self._compile_model()
        print(self.model.summary())
        self.model.fit(
            self.x_train[:500000],
            self.y_train[:500000],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test[:50000],self.y_test[:50000]),
            callbacks=self._save_checkpoint()
            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(self.model_filename+""+".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.model_filename+""+".h5")

            