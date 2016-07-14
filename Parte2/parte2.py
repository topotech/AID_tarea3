# *-* coding: utf-8 *-*

import urllib
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random

# Parte a)

# Importación de datos
#Nota: Esto está comentado para evitar descargar los datos cada vez

train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
train_data_f = urllib.urlretrieve(train_data_url, "polarity.train")
test_data_f = urllib.urlretrieve(test_data_url, "polarity.dev")

#Encabezado y count de la data

ftr = open("polarity.train", "r")
fts = open("polarity.dev", "r")
rows = [line.split(" ", 1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
train_df['Sentiment'] = pd.to_numeric(train_df['Sentiment'])
rows = [line.split(" ", 1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
test_df['Sentiment'] = pd.to_numeric(test_df['Sentiment'])
print "\n a) Forma de los datos: Sentimientos negativos (-1) y positivos (+1)\n"
print "Datos training:"
print train_df['Sentiment'].value_counts()
print "\nDatos test:"
print test_df['Sentiment'].value_counts()
print ""


#Parte b)

#Definición de funciones
def word_preprocessing(text):
    text = text.decode('utf-8', 'ignore')
    text = re.sub(r'([a-z])\1+', r'\1\1', text)
    words = word_tokenize(text)
    commonwords = stopwords.words('english')
    words = [word.lower() \
             for word in words if word.lower() not in commonwords]
    return ' '.join(words)


def word_extractor(text):
    text = text.decode('utf-8', 'ignore')
    words = word_tokenize(text)
    commonwords = stopwords.words('english')

    porter = PorterStemmer()
    words = [porter.stem(word).encode('utf-8') for word in words]

    words = ' '.join(words)
    return words

#Cadenas de prueba
teststring = ["I love to eat cake"
    , "I love eating cake"
    , "I loved eating the cake"
    , "I do not love eating cake"
    , "I don't love eating cake"
    , "Those are stupid dogs"
    , "It wasn't really a baaaad movie"]
teststring_preproc = []

#Output pedido
print "b)\n"

print "Preprocesado SIN stemming:"
for i in range(0,len(teststring),1):
    teststring_preproc.insert(i, word_preprocessing(teststring[i]) )
    print teststring_preproc[i]

print "\nPreprocesado CON stemming:"
for i in range(0,len(teststring),1):
    print word_extractor(teststring_preproc[i])

#Parte c)

#Definición de lematizador
def word_extractor2(text):
    wordlemmatizer = WordNetLemmatizer()
    text = text.decode('utf-8', 'ignore')
    words = word_tokenize(text)

    words = [wordlemmatizer.lemmatize(word) for word in words]
    words = ' '.join(words)
    return words


#Output pedido
print "\nc)\n"

print "Preprocesado CON lematizar:"
for i in range(0,len(teststring),1):
    print word_extractor2(teststring_preproc[i])



#Parte d)

texts_train=[0,0]
texts_test=[0,0]
vectorizer=[0,0]
features_train=[0,0]
features_test=[0,0]
vocab=[0,0]
dist_train=[0,0]
count_train=[0,0]
dist_test=[0,0]
count_test=[0,0]

#Preparacion datos: [0]: Stemming ,[1] Lematizar

labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)


#Switch: 1: sólo hace stemming (se demora menos), 2: procesa además lemmatizer (más costoso)

my_switch = 2

texts_train[0] = [word_extractor(word_preprocessing(text)) for text in train_df.Text]
texts_test[0] = [word_extractor(word_preprocessing(text)) for text in test_df.Text]

if (i >= 2):
    texts_train[1] = [word_extractor2(word_preprocessing(text)) for text in train_df.Text]
    texts_test[1] = [word_extractor2(word_preprocessing(text)) for text in test_df.Text]

print "\nd)\n"

#Contador de palabras post-procesamiento
for i in range(0,my_switch ,1):

    vectorizer[i] = CountVectorizer(ngram_range=(1, 1), binary='False')
    vectorizer[i].fit(np.asarray(texts_train[i]))
    features_train[i] = vectorizer[i].transform(texts_train[i])
    features_test[i] = vectorizer[i].transform(texts_test[i])
    vocab[i] = vectorizer[i].get_feature_names()


    dist_train[i] = list(np.array(features_train[i].sum(axis=0)).reshape(-1,))
    count_train[i] = zip(vocab[i], dist_train[i])
    print "Training data:"
    print sorted(count_train[i], key=lambda x: x[1], reverse=True)[:100]

    dist_test[i] = list(np.array(features_test[i].sum(axis=0)).reshape(-1,))
    count_test[i] = zip(vocab[i], dist_test[i])
    print "Test data:"
    print sorted(count_test[i], key=lambda x: x[1], reverse=True)[:100]



#Parte e)

def score_the_model(model, x, y, xt, yt, text):
    acc_tr = model.score(x, y)
    acc_test = model.score(xt[:-1], yt[:-1])
    print "Training Accuracy %s: %f" % (text, acc_tr)
    print "Test Accuracy %s: %f" % (text, acc_test)
    print "Detailed Analysis Testing Results ..."
    print(classification_report(yt, model.predict(xt), target_names=['+', '-']))


#Parte f)
def do_NAIVE_BAYES(x, y, xt, yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "BernoulliNB")
    return model

model = do_NAIVE_BAYES(features_train, labels_train, features_test, labels_test)
test_pred = model.predict_proba(features_test)
spl = random.sample(xrange(len(test_pred)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
    print sentiment, text


#Parte g)
def do_MULTINOMIAL(x, y, xt, yt):
    model = MultinomialNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "MULTINOMIAL")
    return model

# model = do_MULTINOMIAL(features_train, labels_train, features_test, labels_test)
# test_pred = model.predict_proba(features_test)
# spl = random.sample(xrange(len(test_pred)), 15)
# for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
#     print sentiment, text



#Parte h)
def do_LOGIT(x, y, xt, yt):
    start_t = time.time()
    Cs = [0.01, 0.1, 10, 100, 1000]
    for C in Cs:
        print "Usando C= %f" % C
        model = LogisticRegression(penalty='l2', C=C)
        model = model.fit(x, y)
        score_the_model(model, x, y, xt, yt, "LOGISTIC")

# do_LOGIT(features_train, labels_train, features_test, labels_test)


#Parte i)
def do_SVM(x, y, xt, yt):
    Cs = [0.01, 0.1, 10, 100, 1000]
    for C in Cs:
        print "El valor de C que se esta probando: %f" % C
        model = LinearSVC(C=C)
        model = model.fit(x, y)
        score_the_model(model, x, y, xt, yt, "SVM")


# do_SVM(features_train, labels_train, features_test, labels_test)

#Parte j)
#HERE BE DRAGONS