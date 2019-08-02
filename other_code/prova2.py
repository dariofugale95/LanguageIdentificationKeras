from Enviroment import Enviroment as env
from dataprocessing import DataReader, DataGenerator
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.metrics import confusion_matrix, accuracy_score
from metrics import print_confusion_matrix
import numpy as np
import pickle

def load_data(): 
    with open(env().path_to_arrays, "rb") as f_pickle:
        X_tr = pickle.load(f_pickle)
        X_te = pickle.load(f_pickle)
        y_tr = pickle.load(f_pickle)
        y_te = pickle.load(f_pickle)
        max_words = pickle.load(f_pickle)
    return X_tr, X_te, y_tr, y_te, max_words

dr = DataReader(env().data_dir)

# data, labels = dr.create_dataset()
# dr.save_data_csv()
# data, labels = dr.load_dataset()
# dg = DataGenerator(data, labels)
# data, labels = dg.generate_data()

# X_train, X_test, y_train, y_test = dg.split_train_test(env().DIM_TEST)

# dg.save_data(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test, max_words_length = load_data()

y_tr, y_te = [],[]
for lab in y_train:
    y_tr.append(np.argmax(lab))
for lab in y_test:
    y_te.append(np.argmax(lab))

y_train = y_tr
y_test = y_te

print(X_train[:40])
print(y_train[:40])
print(y_test[:40])
classifier = NB()
print("Fitting NB model\n")
classifier.fit(X_train, y_train)

# Predict Class
print("Predict class\n")
y_pred = classifier.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test,y_pred)
print("Accurcay %f" %accuracy)

np.save("naivebayesclassifier",classifier)
#sentence = "ciao sono dario e ho ventiquattro anni, posto molto bello da vedere"
#y_sentence_predicted = predictLanguage(sentence, classifier)
#print(y_sentence_predicted)