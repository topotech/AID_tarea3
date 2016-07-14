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
#from sklearn.svm import LinearSVC
from prob_svm import LinearSVC_proba as LinearSVC
import random
import matplotlib.pyplot as plt




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

texts_train=[0,0,0,0]
texts_test=[0,0,0,0]
vectorizer=[0,0,0,0]
features_train=[0,0,0,0]
features_test=[0,0,0,0]
vocab=[0,0,0,0]
dist_train=[0,0,0,0]
count_train=[0,0,0,0]
dist_test=[0,0,0,0]
count_test=[0,0,0,0]

#Preparacion datos: [0]: Stemming , [1]: Stemming con preproc ,[2] Lematizar, [3] Lematizar con preproc

labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)


#Switch: 2: sólo hace stemming (se demora menos), 4: procesa además lemmatizer (más costoso)


my_switch = 4

texts_train[0] = [word_extractor(text) for text in train_df.Text]
texts_test[0] = [word_extractor(text) for text in test_df.Text]
texts_train[1] = [word_extractor(word_preprocessing(text)) for text in train_df.Text]
texts_test[1] = [word_extractor(word_preprocessing(text)) for text in test_df.Text]


if (my_switch >= 3):
    texts_train[2] = [word_extractor2(text) for text in train_df.Text]
    texts_test[2] = [word_extractor2(text) for text in test_df.Text]
    texts_train[3] = [word_extractor2(word_preprocessing(text)) for text in train_df.Text]
    texts_test[3] = [word_extractor2(word_preprocessing(text)) for text in test_df.Text]

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

    if (i==0):
        print "Top10 palabras:STEMMING:\n"
    if (i==1):
        print "Top10 palabras:STEMMING SIN STOPWORDS (PREPROCESADO):\n"
    if (i==2):
        print "Top10 palabras:LEMMATIZING:\n"
    if (i==3):
        print "Top10 palabras: LEMMATIZING SIN STOPWORDS (PREPROCESADO):\n"

    print "\tTraining data:"
    print "\t"+str(sorted(count_train[i], key=lambda x: x[1], reverse=True)[:100])

    dist_test[i] = list(np.array(features_test[i].sum(axis=0)).reshape(-1,))
    count_test[i] = zip(vocab[i], dist_test[i])
    print "\tTest data:"
    print "\t"+str(sorted(count_test[i], key=lambda x: x[1], reverse=True)[:100])



#Parte e)

global_accuracies=[]; #<-Me odio por esto

print "\n /*Parte e) no imprime nada*/ \n"
def score_the_model(model, x, y, xt, yt, text):
    acc_tr = model.score(x, y)
    acc_test = model.score(xt[:-1], yt[:-1])
    print "Training Accuracy %s: %f" % (text, acc_tr)
    print "Test Accuracy %s: %f" % (text, acc_test)
    print "Detailed Analysis Testing Results ..."
    print(classification_report(yt, model.predict(xt), target_names=['+', '-']))
    global_accuracies.append((acc_tr,acc_test)) # Cada vez que esto se ejecuta muere un gatito. 48 gatitos muetos :C

#Parte f)
def do_NAIVE_BAYES(x, y, xt, yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "BernoulliNB")
    return model

print "\n f) Naive Bayes \n"
for i in range(0,my_switch ,1):
    if (i==0):
        print "STEMMING:\n"
    if (i==1):
        print "STEMMING SIN STOPWORDS (PREPROCESADO):\n"
    if (i==2):
        print "LEMMATIZING:\n"
    if (i==3):
        print "LEMMATIZING SIN STOPWORDS (PREPROCESADO):\n"
    model = do_NAIVE_BAYES(features_train[i], labels_train, features_test[i], labels_test)
    test_pred = model.predict_proba(features_test[i])
    spl = random.sample(xrange(len(test_pred)), 15)
    for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
        print sentiment, text

#Parte g)
def do_MULTINOMIAL(x, y, xt, yt):
    model = MultinomialNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "MULTINOMIAL")
    return model

print "\n g) Naive Bayes Multinomial\n"
for i in range(0,my_switch ,1):
    if (i==0):
        print "STEMMING:\n"
    if (i==1):
        print "STEMMING SIN STOPWORDS (PREPROCESADO):\n"
    if (i==2):
        print "LEMMATIZING:\n"
    if (i==3):
        print "LEMMATIZING SIN STOPWORDS (PREPROCESADO):\n"
    model = do_MULTINOMIAL(features_train[i], labels_train, features_test[i], labels_test)
    test_pred = model.predict_proba(features_test[i])
    spl = random.sample(xrange(len(test_pred)), 15)
    for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
        print sentiment, text


#Parte h)
Cs = [0.01, 0.1, 10, 100, 1000]

def do_LOGIT(x, y, xt, yt):
    start_t = time.time()
    Cs = [0.01, 0.1, 10, 100, 1000]
    model = []
    for i in range(0,len(Cs),1):
        print "Usando C= %f" % Cs[i]
        model.append ( LogisticRegression(penalty='l2', C=Cs[i]) )
        model[i] = model[i].fit(x, y)
        score_the_model(model[i], x, y, xt, yt, "LOGISTIC")
    return model

print "\n h) Regresión logística regularizada con penalizador norma l_2\n"
for i in range(0,my_switch ,1):
    if (i==0):
        print "STEMMING:\n"
    if (i==1):
        print "STEMMING SIN STOPWORDS (PREPROCESADO):\n"
    if (i==2):
        print "LEMMATIZING:\n"
    if (i==3):
        print "LEMMATIZING SIN STOPWORDS (PREPROCESADO):\n"

    models = do_LOGIT(features_train[i], labels_train, features_test[i], labels_test)

    for j, model in enumerate(models):
        print "\tC = "+str(Cs[j])+" :\n"
        test_pred = model.predict_proba(features_test[i])
        spl = random.sample(xrange(len(test_pred)), 15)
        for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
            print sentiment, text


#Parte i)
def do_SVM(x, y, xt, yt):
    Cs = [0.01, 0.1, 10, 100, 1000]
    model = []
    for i in range (0,len(Cs),1):
        print "El valor de C que se esta probando: %f" % Cs[i]
        model.append( LinearSVC(C=Cs[i]) )
        model[i] = model[i].fit(x, y)
        score_the_model(model[i], x, y, xt, yt, "SVM")
    return model

print "\n i) Support Vector Machine\n"
for i in range(0,my_switch ,1):
    if (i==0):
        print "STEMMING:\n"
    if (i==1):
        print "STEMMING SIN STOPWORDS (PREPROCESADO):\n"
    if (i==2):
        print "LEMMATIZING:\n"
    if (i==3):
        print "LEMMATIZING SIN STOPWORDS (PREPROCESADO):\n"

    models = do_SVM(features_train[i], labels_train, features_test[i], labels_test)

    for j, model in enumerate(models):
        print "\tC = "+str(Cs[j])+" :\n"
        test_pred = model.predict_proba(features_test[i])
        spl = random.sample(xrange(len(test_pred)), 15)
        for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
            print sentiment, text

#Parte j)
plot_labels = [
      "NB_S"
    , "NB_SP"
    , "NB_L"
    , "NB_LP"
    , "MN_S"
    , "MN_SP"
    , "MN_L"
    , "MN_LP"
    , "LO_S_-2"
    , "LO_S_-1"
    , "LO_S_+1"
    , "LO_S_+2"
    , "LO_S_+3"
    , "LO_SP_-2"
    , "LO_SP_-1"
    , "LO_SP_+1"
    , "LO_SP_+2"
    , "LO_SP_+3"
    , "LO_L_-2"
    , "LO_L_-1"
    , "LO_L_+1"
    , "LO_L_+2"
    , "LO_L_+3"
    , "LO_LP_-2"
    , "LO_LP_-1"
    , "LO_LP_+1"
    , "LO_LP_+2"
    , "LO_LP_+3"
    , "SV_S_-2"
    , "SV_S_-1"
    , "SV_S_+1"
    , "SV_S_+2"
    , "SV_S_+3"
    , "SV_SP_-2"
    , "SV_SP_-1"
    , "SV_SP_+1"
    , "SV_SP_+2"
    , "SV_SP_+3"
    , "SV_L_-2"
    , "SV_L_-1"
    , "SV_L_+1"
    , "SV_L_+2"
    , "SV_L_+3"
    , "SV_LP_-2"
    , "SV_LP_-1"
    , "SV_LP_+1"
    , "SV_LP_+2"
    , "SV_LP_+3"
]

tr_accuracy = [item[0] for item in global_accuracies]
test_accuracy = [item[1] for item in global_accuracies]

ayuda=[]
for i in range(1,len(test_accuracy)+1):
	ayuda.append(i)




plt.plot( ayuda, test_accuracy)
plt.plot( ayuda, tr_accuracy)
plt.xticks(ayuda, plot_labels, rotation=90)
plt.legend(['Test accuracies', 'Training Accuracies'], loc='upper left')
plt.show()
