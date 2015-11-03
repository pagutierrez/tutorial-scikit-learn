# Tutorial sobre `scikit-learn`
Este breve tutorial explica algunos de los conceptos relacionados con la librería `scikit-learn` de python. 

# ¿Qué es python?

- Python es un lenguaje de programación interpretado.
- Su nombre proviene de la afición de su creador original, [Guido van Rossum](https://es.wikipedia.org/wiki/Guido_van_Rossum), por los humoristas británicos [Monty Python](https://es.wikipedia.org/wiki/Monty_Python).
- Características:
  - Programación orientada a objetos
  - Programación imperativa
  - Programación funcional.
  - Es multiplataforma y posee una licencia abierta.

# Entornos de desarrollo para python

- Entornos de desarrollo para Python
  - [Sublime Text](http://www.sublimetext.com/)
  - [PyCharm](https://www.jetbrains.com/pycharm/)
  - [Spyder](https://github.com/spyder-ide/spyder)

# `scikit-learn`

- Librería que proporciona un amplio conjunto de algoritmos de aprendizaje supervisado y no supervisado a través de una consistente interfaz en `python`.
- Publicado bajo licencia BSD y distribuido en muchos sistemas Linux, favorece el uso comercial y educacional.
- Esta librería se ha construido sobre [`SciPy`](http://www.scipy.org/) (*Scientific Python*), que debe ser instalada antes de utilizarse, incluyendo:
  - [**NumPy**](http://www.numpy.org/)
  - [**Matplotlib**](http://matplotlib.org/)
  - [SymPy](https://simpy.readthedocs.org/en/latest/)
  - [**Pandas**](http://pandas.pydata.org/)


# Características de `scikit-learn`

- Esta librería se centra en el modelado de datos y no en cargar y manipular los dato, para lo que utilizaríamos [NumPy](http://www.numpy.org/) y [Pandas](http://pandas.pydata.org/). Algunas cosas que podemos hacer con `scikit-learn` son:
  - *Clustering*.
  - Validación cruzada.
  - *Datasets* de prueba.
  - Reducción de la dimensionalidad.
  - *Ensemble methods*.
  - *Feature selection*.
  - *Parameter tuning*.

Las principales ventajas de `scikit-learn` son las siguientes:
  - Interfaz consistente ante modelos de aprendizaje automático.
  - Proporciona muchos parámetros de configuración.
  - Documentación excepcional.
  - Desarrollo muy activo.
  - Comunidad.

# Ejemplo de uso con el *dataset* `iris`

Vamos a utilizar un ejemplo típico en *machine learning* que es la base de datos `iris`.  En esta base de datos hay tres clases a predecir, que son tres especies distintas de la flor iris, de manera que, para cada flor, se extraen cuatro medidas o variables de entrada (longitud y ancho de los pétalos y los sépalos, en cm). Las tres especies a distinguir son iris *setosa*, iris *virginica* e iris *versicolor*.

## Lectura de datos

Como ya hemos comentado, para la lectura de datos haremos uso de [Pandas](http://pandas.pydata.org/). Esta librería tiene un método `read_csv` que nos va a permitir leer los datos desde un fichero de texto `csv`. Veamos el código:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn import preprocessing
```
Con estas líneas, importamos la funcionalidad necesaria para el ejemplo. `pandas` nos permitirá leer los datos, `numpy` nos va a permitir trabajar con ellos de forma matricial, `matplotlib` nos permite hacer representaciones gráficas y, de la librería `scikit-learn`, en este caso, utilizaremos un método de clasificación basado en los vecinos más cercanos y algunas funciones de preprocesamiento.

El método `read_csv` de `pandas` permite dos modos de trabajo: que el propio fichero csv tenga una fila con los nombres de las variables o que nosotros especifiquemos los nombres de las variables en la llamada. En este caso, vamos a utilizar la segunda aproximación. De esta forma, creamos un *array* con los nombres de las variables:
```python
nombre_variables = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']
```
y leemos el dataset con:
```python
iris = pd.read_csv('data/iris.csv', names = nombre_variables)
```
`iris` es un objeto de la clase [`DataFrame`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) de `pandas`.

## Inspección de datos

Antes de nada, es conveniente realizar una pequeña **inspección** de los datos. Si simplemente queremos ver la cabecera del dataset, podemos utilizar el método `head(n)`, que devuelve un DataFrame incluyendo los primeros `n` patrones:
```python
print iris.head(9)
```
Ahora vamos a utilizar una función para inspeccionar detenidamente cada par de variables y su relación con las etiquetas de clase. De esta forma, construiremos un gráfico de (3x4) subgráficos, que incluya, para cada par de variables, los 150 patrones, con un calor que indique la etiqueta de clase y donde las coordenadas x e y se correspondan con los valores de las variables afectadas. Esto se puede hacer con el siguiente código:
```python
def plot_dataset(dataset,nombre_variables):
    """ Función que pinta un dataset
    dataset es el DataFrame que vamos a utilizar para extraer los datos
    nombre_variables es el nombre de las variables de ese dataset
    * Supondremos que la última variable es la etiqueta de clase
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
```

Si ahora queremos usar la función con el dataset iris, podemos llamarla de la siguiente forma:
```python
plot_dataset(iris,nombre_variables)
```
El resultado debería ser el siguiente:
![Scatterplot de iris](images/iris.png)

# Referencias
- Python como alternativa a R en *machine learning*. Mario Pérez Esteso. [Enlace a Github](https://github.com/MarioPerezEsteso/Python-Machine-Learning). [Enlace a Youtube](https://www.youtube.com/watch?v=8yz4gWt7Klk). 
- *An introduction to machine learning with scikit-learn*. Documentación oficial de `scikit-learn`. [http://scikit-learn.org/stable/tutorial/basic/tutorial.html](http://scikit-learn.org/stable/tutorial/basic/tutorial.html).
- *A tutorial on statistical-learning for scientific data processing*. Documentación oficial de `scikit-learn`. [http://scikit-learn.org/stable/tutorial/statistical_inference/index.html](http://scikit-learn.org/stable/tutorial/statistical_inference/index.html).