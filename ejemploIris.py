#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:44:59 2015

@author: pagutierrez
"""

# Librerías a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import neighbors


# Nombre de las variables
nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 
                    'ancho_petalo', 'clase']

#Lectura desde un fichero CSV
iris = pd.read_csv('data/iris.csv', names = nombre_variables)
print type(iris)

# Imprimir el DataFrame de los 9 primeros patrones
print iris.head(9)

def plot_dataset(dataset,nombre_variables):
    """ 
    Función que pinta un dataset. Supondremos que la última variable es la 
    etiqueta de clase
    Recibe los siguientes argumentos:
    - dataset: DataFrame que vamos a utilizar para extraer los datos
    - nombre_variables: array con el nombre de las variables de ese dataset
    Devuelve: NADA
    """
    # Numero de variables
    num_variables = dataset.shape[1]
    # Extraer la etiqueta de clase
    labels = dataset[nombre_variables[-1]]
    # Convertir la etiqueta a números enteros (1,2,3...)
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(labels)
    labels = labelencoder.transform(labels)
    # Número de plot
    plot_index = 1
    plt.figure(figsize=(18,12))
    plt.clf()
    for i in range(0,num_variables-1):
        for j in range(0,num_variables-1):
            if i != j:  
                # Extraer variables i y j
                x = dataset[nombre_variables[i]]
                y = dataset[nombre_variables[j]]
                # Elegir el subplot
                plt.subplot(num_variables-2, num_variables-1, plot_index)
                # Pintar los puntos
                plt.scatter(x, y, c=labels)
                # Etiquetas para los ejes
                plt.xlabel(nombre_variables[i])
                plt.ylabel(nombre_variables[j])
                # Título para el gráfico
                plt.title(nombre_variables[j]+" vs "+nombre_variables[i])
                # Extraer rangos de las variables y establecerlos
                x_min, x_max = x.min() - .5, x.max() + .5
                y_min, y_max = y.min() - .5, y.max() + .5
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                # Que no se vean los ticks
                plt.xticks(())
                plt.yticks(())
                plot_index = plot_index + 1
    plt.show()


plot_dataset(iris,nombre_variables)

# Generar un array con el área del sépalo (longitud*anchura), utilizando un for
# Crear un array vacío
iris_array = iris.values
areaSepaloArray = np.empty(iris_array.shape[0]) # OJO paréntesis
for i in range(0,iris_array.shape[0]):
    areaSepaloArray[i] = iris_array[i,0] * iris_array[i,1]
print areaSepaloArray

# Generar un array con el área del sépalo (longitud*anchura), utilizando operaciones matriciales
# Crear un array vacío e inicializarlo a 0
print iris_array[:,0] * iris_array[:,1]

# Imprimir las longitudes de sépalo mayores que 2, utilizando un for
iris_array = iris.values
for i in range(0,iris_array.shape[0]):
    valorSepalo = iris_array[i,0]
    if valorSepalo > 2:
        print valorSepalo
        
# Imprimir las longitudes de sépalo mayores que 2, utilizando operaciones matriciales
print iris_array[ iris_array[:,0] > 2, 0]


def dividir_ent_test(dataframe, porcentaje=0.6):
    """ 
    Función que divide un dataframe aleatoriamente en entrenamiento y en test.
    Recibe los siguientes argumentos:
    - dataframe: DataFrame que vamos a utilizar para extraer los datos
    - porcentaje: porcentaje de patrones en entrenamiento
    Devuelve:
    - train: DataFrame con los datos de entrenamiento
    - test: DataFrame con los datos de test
    """
    mascara = np.random.rand(len(dataframe)) < porcentaje
    train = dataframe[mascara]
    test = dataframe[~mascara]
    return train, test

# Dividimos el dataset
iris_train, iris_test = dividir_ent_test(iris)

# Extraer entradas y salidas en entrenamiento (-1 es la última columna)
train_inputs_iris = iris_train.values[:,0:-1]
train_outputs_iris = iris_train.values[:,-1]

# Extraer entradas y salidas en test (-1 es la última columna)
test_inputs_iris = iris_test.values[:,0:-1]
test_outputs_iris = iris_test.values[:,-1]
print train_inputs_iris.shape

# LabelEncoder -> Transforma etiquetas en números enteros
label_e = preprocessing.LabelEncoder()
label_e.fit(train_outputs_iris)
train_outputs_iris_encoded = label_e.transform(train_outputs_iris)
test_outputs_iris_encoded = label_e.transform(test_outputs_iris)
print label_e.classes_

# Crear un knn y entrenarlo
knn = neighbors.KNeighborsClassifier()
knn.fit(train_inputs_iris, train_outputs_iris_encoded)
print knn

# Predecir el fichero de test
prediccion_test = knn.predict(test_inputs_iris)
print prediccion_test

# Comprobar la precisión del clasificador en test
precision = knn.score(test_inputs_iris, test_outputs_iris_encoded)
print "Valor de precisión en test: %.2f" % precision

# Crear una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_outputs_iris_encoded, prediccion_test)
print cm

# Probar distintos valores del parámetro número de vecinos (de 1 a 14)
for nn in range(1,15):
    knn = neighbors.KNeighborsClassifier(n_neighbors=nn)
    knn.fit(train_inputs_iris, train_outputs_iris_encoded)
    precisionTrain = knn.score(train_inputs_iris, train_outputs_iris_encoded)
    precisionTest = knn.score(test_inputs_iris, test_outputs_iris_encoded)
    print "%d vecinos: CCR train=%.2f%%, CCR test=%.2f%%" % \
        (nn, precisionTrain*100, precisionTest*100)