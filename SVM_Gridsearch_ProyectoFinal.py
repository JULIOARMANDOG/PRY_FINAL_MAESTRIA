#IMPORTACION DE LIBRERIAS
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
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib

#Importacion de datos
X_train_route = 'uci-har/train/X_train.txt'
y_train_route = 'uci-har/train/y_train.txt'
X_train = pd.read_csv(X_train_route, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_route, delim_whitespace=True, header=None)

#transformar y_train a un array de numpy
y_train = y_train.values.ravel()
# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#definimos la cuadricula de hiperparámetros
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
# Crear el objeto GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
# Entrenar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)
# Imprimir los mejores parámetros encontrados
print("Mejores parámetros encontrados:")
print(grid_search.best_params_)
# Imprimir la mejor puntuación
print("Mejor puntuación:")
print(grid_search.best_score_)
# Obtener el mejor modelo
best_model = grid_search.best_estimator_
# Evaluar el modelo SVM
X_test_route = 'uci-har/test/X_test.txt'
y_test_route = 'uci-har/test/y_test.txt'
X_test = pd.read_csv(X_test_route, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_route, delim_whitespace=True, header=None)
# Convertir y test a un array de numpy
y_test = y_test.values.ravel()
# Normalizar los datos de prueba
X_test_scaled = scaler.transform(X_test)
# Predecir con el mejor modelo SVM
y_pred = best_model.predict(X_test_scaled)
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

ruta_salida = os.path.join("assets", "svm", "confusion.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')

plt.show()
# Grafico de curva ROC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Binarizar las etiquetas
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test))
# Calcular la curva ROC
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)
# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
ruta_salida = os.path.join("assets", "svm", "roc.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()


joblib.dump(best_model, "modelo_svm_gridsearch.pkl")
joblib.dump(scaler, "scaler_svm.pkl")
print("\n✅ Modelo y scaler guardados con éxito.")

