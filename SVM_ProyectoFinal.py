#iMPORTACION DE LIBRERIAS 
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from collections import Counter
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.svm import SVC
import seaborn as sns


# CARGA DE DATOS
X_train_route = 'sample_data/data/uci-har/train/X_train.txt'
y_train_route = 'sample_data/data/uci-har/train/y_train.txt'

X_train = pd.read_csv(X_train_route, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_route, delim_whitespace=True, header=None)

# Convertir y train a un array de numpy
y_train = y_train.values.ravel()
# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Definir el modelo SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=44)
# Entrenar el modelo SVM
svm_model.fit(X_train, y_train)

# Evaluar el modelo SVM
X_test_route = 'sample_data/data/uci-har/test/X_test.txt'
y_test_route = 'sample_data/data/uci-har/test/y_test.txt'

X_test = pd.read_csv(X_test_route, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_route, delim_whitespace=True, header=None)
# Convertir y test a un array de numpy
y_test = y_test.values.ravel()

# Normalizar los datos de prueba
X_test_scaled = scaler.transform(X_test)
# Predecir con el modelo SVM
y_pred = svm_model.predict(X_test_scaled)
# Imprimir el reporte de clasificación
print(classification_report(y_test, y_pred))
# Imprimir la matriz de confusión
print(confusion_matrix(y_test, y_pred))
# Graficar la matriz de confusión
labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()

