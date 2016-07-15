import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error 

#Pregunta A
#Se descargan los datasets como csv, desde su direccion original
train_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
test_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.test"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
#Se transforman los datasets en dataframes
train_df = pd.DataFrame.from_csv('train_data.csv',header=0,index_col=0)
test_df = pd.DataFrame.from_csv('test_data.csv',header=0,index_col=0)

#Pregunta B
#Se genera la matriz de datos a utilizar, y el vector de respuestas
#Se obvia la columna de id registro pues no aporta al analisis
X = train_df.ix[:,'x.1':'x.10'].values
y = train_df.ix[:,'y'].values
X_std = StandardScaler().fit_transform(X)

#Pregunta C
sklearn_pca = PCA(n_components=2)
Xred_pca = sklearn_pca.fit_transform(X_std)
cmap = plt.cm.get_cmap('gist_rainbow')
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_pca[y==lab, 0],Xred_pca[y==lab, 1],label=lab,c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()


#Pregunta D
sklearn_lda = LinearDiscriminantAnalysis(n_components=2)
Xred_lda = sklearn_lda.fit_transform(X_std,y)
cmap = plt.cm.get_cmap('gist_rainbow')
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses,mcolors):
    plt.scatter(Xred_lda[y==lab, 0],Xred_lda[y==lab, 1],label=lab,c=col)
plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()

#Pregunta E: En informe

#Pregunta F
probs = []
count = 0.0
for i in range(1,11):
	for j in range (0,y.shape[0]):
		if(i == y[j]):
			count = count + 1
	probs.append(round(float(count/y.shape[0]),4))
	count = 0.0

#probabilidad dado y = clase
def a_priori(y,clase):
	print "\nEjemplo Clasificador:\nProbabilidad de X dada Clase = %d" % clase
	return y[clase]

#para llamar al clasificador se utiliza el arreglo de probabilidades calculado
print a_priori(probs, 3)
print "\n"
#Pregunta G
Xtest = test_df.ix[:,'x.1':'x.10'].values
ytest = test_df.ix[:,'y'].values
X_std_test = StandardScaler().fit_transform(Xtest)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_std,y)
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_std,y)
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_std,y)
print "LDA_Score TRAIN: %.4f" % lda_model.score(X_std,y)
print "LDA_Score TEST: %.4f" % lda_model.score(X_std_test,ytest)
print "QDA_Score TRAIN: %.4f" % qda_model.score(X_std,y)
print "QDA_Score TEST: %.4f" % qda_model.score(X_std_test,ytest)
print "KNN10_Score TRAIN: %.4f" % knn_model.score(X_std,y)
print "KNN10_Score TEST: %.4f" % knn_model.score(X_std_test,ytest)


scores_train = []
scores_test = []
for k in range(1,32):
	knn_model = KNeighborsClassifier(n_neighbors = k)
	knn_model.fit(X_std,y)
	scores_train.append(knn_model.score(X_std,y))
	scores_test.append(knn_model.score(X_std_test,ytest))

plt.figure(figsize=(12, 8))
plt.plot(scores_train, label='Score de Entrenamiento')
plt.plot(scores_test, label='Score de Prueba')
plt.xlabel('K Vecinos')
plt.ylabel('Score KNN')
leg = plt.legend(loc='upper right', fancybox=True)
plt.show()

#Pregunta H

error_ldatrain = []
error_ldatest = []
error_qdatrain = []
error_qdatest = []
error_knntrain = []
error_knntest = []

lda_model = LinearDiscriminantAnalysis()
qda_model = QuadraticDiscriminantAnalysis()
knn_model = KNeighborsClassifier(n_neighbors=10)
for d in range(1,11):
	#Se reduce a d dimensiones 
	sklearn_pca = PCA(n_components=d)
	Xred_pca = sklearn_pca.fit_transform(X_std)
	Xred_pcatest = sklearn_pca.transform(X_std_test);
	#Se ajusta a los distintos clasificadores	
	lda_model.fit(Xred_pca,y)
	qda_model.fit(Xred_pca,y)
	knn_model.fit(Xred_pca,y)
	
	#Se calculan los errores de entrenamiento y prueba para cada caso
	error_ldatrain.append(1-lda_model.score(Xred_pca,y))
	error_ldatest.append(1-lda_model.score(Xred_pcatest,ytest))
	error_qdatrain.append(1-qda_model.score(Xred_pca,y))
	error_qdatest.append(1-qda_model.score(Xred_pcatest,ytest))
	error_knntrain.append(1-knn_model.score(Xred_pca,y))
	error_knntest.append(1-knn_model.score(Xred_pcatest,ytest))

plt.figure(figsize=(12,8))
plt.plot(error_ldatrain, label="LDA_Entrenamiento")
plt.plot(error_ldatest, label="LDA_Prueba")
plt.plot(error_qdatrain, label="LDA_Entrenamiento")
plt.plot(error_qdatest, label="QDA_Prueba")
plt.plot(error_knntrain, label="LDA_Entrenamiento")
plt.plot(error_knntest, label="KNN_Prueba")
plt.xlabel("d dimensiones (PCA)")
plt.ylabel("Error")
plt.legend(loc="right", fancybox = True)
plt.show()


#Pregunta I

error_ldatrain = []
error_ldatest = []
error_qdatrain = []
error_qdatest = []
error_knntrain = []
error_knntest = []

lda_model = LinearDiscriminantAnalysis()
qda_model = QuadraticDiscriminantAnalysis()
knn_model = KNeighborsClassifier(n_neighbors=10)
for d in range(1,11):
	#Se reduce a d dimensiones 
	sklearn_lda = LinearDiscriminantAnalysis(n_components=d)
	Xred_lda = sklearn_lda.fit_transform(X_std, y)
	Xred_ldatest = sklearn_lda.transform(X_std_test);
	#Se ajusta a los distintos clasificadores	
	lda_model.fit(Xred_lda,y)
	qda_model.fit(Xred_lda,y)
	knn_model.fit(Xred_lda,y)
	
	#Se calculan los errores de entrenamiento y prueba para cada caso
	error_ldatrain.append(1-lda_model.score(Xred_lda,y))
	error_ldatest.append(1-lda_model.score(Xred_ldatest,ytest))
	error_qdatrain.append(1-qda_model.score(Xred_lda,y))
	error_qdatest.append(1-qda_model.score(Xred_ldatest,ytest))
	error_knntrain.append(1-knn_model.score(Xred_lda,y))
	error_knntest.append(1-knn_model.score(Xred_ldatest,ytest))

plt.figure(figsize=(12,8))
plt.plot(error_ldatrain, label="LDA_Entrenamiento")
plt.plot(error_ldatest, label="LDA_Prueba")
plt.plot(error_qdatrain, label="LDA_Entrenamiento")
plt.plot(error_qdatest, label="QDA_Prueba")
plt.plot(error_knntrain, label="LDA_Entrenamiento")
plt.plot(error_knntest, label="KNN_Prueba")
plt.xlabel("d dimensiones (LDA)")
plt.ylabel("Error")
plt.legend(loc="right", fancybox = True)
plt.show()


