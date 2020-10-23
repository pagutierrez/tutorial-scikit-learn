* [Español](#tutorial-sobre-scikit-learn-para-imc)
* [English version](#scikit-learn-tutorial-for-imc)

# Tutorial sobre `scikit-learn` para IMC
Este breve tutorial explica algunos de los conceptos relacionados con la librería `scikit-learn` de python.

Se incluyen dos versiones:

- [Cuaderno Jupyter](tutorial.ipynb)
- [Versión estática](tutorial.md)

Para ejecutar el código tendrás que instalar la última versión de `scikit-learn`. Para la tercera práctica, deberás tener la última versión de `click`:
```bash
pip install click --user --upgrade
pip install scikit-learn --user --upgrade
```
Nota: si haces este tutorial fuera del entorno de la UCO, no es necesario utilizar la opción `--user`, ya que ésta sirve para instalar las bibliotecas en la carpeta de usuario, en lugar de hacerlo a nivel global (que requiere privilegios de administración).

Si no tienes instalado `jupyter` puedes instalarlo con:
```bash
pip install click --user --jupyter
```
También puedes instalar `jupyter`, `click` y `scikit-learn` a través de `apt`, pero las versiones disponibles puede que no sean las últimas.

Lo primero que deberías hacer es clonar el repositorio:
```bash
git clone https://github.com/ayrna/tutorial-scikit-learn-IMC.git
```

Ahora, desde la carpeta del repositorio, lanza `jupyter` mediante:
```bash
cd tutorial-scikit-learn-IMC/
jupyter notebook
```

Si no aparece de forma automática, puede que tengas que abrir una navegador e introducir la dirección indicada (que será del tipo `http://localhost:8888/?token=...`). Una vez iniciada la interfaz de Jupyter abre el fichero `tutorial.ipynb`.

Si estáis usando vuestro portátil, existen instrucciones de instalación de Python con las librerías científicas en:
- [Tutorial completo sobre scikit-learn](https://github.com/ayrna/taller-sklearn-asl-2019)

En cualquier caso, si tenéis cualquier problema con Jupyter, podéis acceder a la [salida del tutorial ya ejecutado](tutorial.md).

Te recomiendo que sigas el [tutorial completo](https://github.com/ayrna/taller-sklearn-asl-2019) para aprender más sobre `scikit-learn`.

# `scikit-learn` tutorial for IMC
This brief tutorial explains some concepts related with the Python library `scikit-learn`.

Two versions are included:

- [Jupyter notebook](tutorialEn.ipynb)
- [Static version](tutorialEn.md)

To run the code, you will need the last version of `scikit-learn`. For the third lab assignment, you will need the last version of `click`:
```bash
pip install click --user --upgrade
pip install scikit-learn --user --upgrade
```
Note: if you run this tutorial from your computer, you do not really need the option `--user`, given that this is for installing libraries for your user instead of system-wide (which requires root privileges).

If you do not have installed `jupyter`, you can install it by:
```bash
pip install click --user --jupyter
```
You can also install `jupyter`, `click` and `scikit-learn` using `apt`, but the available versions may not be the most recent ones.

First, you should clone the repository:
```bash
git clone https://github.com/ayrna/tutorial-scikit-learn-IMC.git
```

Now, from the repository folder, you should run `jupyter` by:
```bash
cd tutorial-scikit-learn-IMC/
jupyter notebook
```

If your browser is not automatically opened, you should manually open it and introduce the address (something similar to `http://localhost:8888/?token=...`). Once you can see the Jupyter server, open the file `tutorialEn.ipynb`.

If you are using your own computer, you may have problems with the scientific libraries. Take a look at:
- [Complete tutorial about scikit-learn](https://github.com/amueller/scipy-2018-sklearn)

In any case, if you are experiencing problems with Jupyter, use the [output of the tutorial](tutorialEn.md).

It is recommended to follow the [complete tutorial](https://github.com/amueller/scipy-2018-sklearn) to learn more about `scikit-learn`.
