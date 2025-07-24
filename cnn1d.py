import requests
import getpass
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot  as plt
from wordcloud import WordCloud
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os
import random

SEED = 45

# Fijar seeds para reproducibilidad
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#token_pat ="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"

# URL RAW del archivo zip
url_git = "https://raw.githubusercontent.com/JULIOARMANDOG/PRY_FINAL_MAESTRIA/main/uci-har.zip"
path_to_data_uci_har="sample_data/data"
message_download_ok="Archivo ZIP descargado correctamente"
message_donwload_into_path="Archivo extraído en carpeta 'uci-har"
message_download_error=" Error al descargar archivo:"

# Crear cabecera donde se incluya el token PAT para la descarga del archivo zip
#headers = {"Authorization": f"token {token_pat}"}
#response_git = requests.get(url_git, headers=headers)
response_git = requests.get(url_git)

if response_git.status_code == 200:
    print(f"✅{message_download_ok}")

    # Extraer zip en memoria y guardar contenido en carpeta 'uci-har'
    with zipfile.ZipFile(io.BytesIO(response_git.content)) as z:
        z.extractall(path_to_data_uci_har)
        print(f"✅ {message_donwload_into_path}")
else:
    print(f"❌ {message_download_error} {response_git.status_code}")
    print(response_git.text)

#Carga data de entrenamiento y test desde el UCI-HAR
df_X_train=pd.read_csv("./sample_data/data/uci-har/train/X_train.txt",sep=r'\s+',header=None)  #pd.read_csv("./sample_data/data/uci-har/train/X_train.txt",sep=r'\s+',header=None)
df_y_train=pd.read_csv("./sample_data/data/uci-har/train/y_train.txt",sep=r'\s+',header=None,names=["activity_id"])
#Cargar los datos de la identificacion de actividades a predecir
df_activities=pd.read_csv("./sample_data/data/uci-har/activity_labels.txt",sep=r'\s+',header=None,names=["activity_id", "activity_label"])
#Para cada valor de la variable objetivo , vamos a asignarle una etiqueta
df_y_train_with_actions=df_y_train.merge(df_activities,on="activity_id")
#cargarmos las etiquetas de las variables predictoras
df_features=pd.read_csv("./sample_data/data/uci-har/features.txt",sep=r'\s+',header=None,names=["index", "feature_name"])
feature_names = pd.Series(df_features["feature_name"])
feature_names = feature_names.where(~feature_names.duplicated(), feature_names + "_" + feature_names.groupby(feature_names).cumcount().astype(str))
df_X_train.columns = feature_names

#Registramos el numero de veces que una actividad se repite dentro del dataset
#Devolviendo una serie en la cual la actividad se identifica como indice y el
#numero de ocurriencias de esa actividad como valor ; y ordenando ese resultado
#según el indice alfabetico o numérico
#num_activities=df_y_train["activity_label"].value_counts().sort_index()
#print(df_activities.head())
#Cargamos el dataSet de prueba
df_X_test=pd.read_csv("./sample_data/data/uci-har/test/X_test.txt",sep=r'\s+',header=None)
df_y_test=pd.read_csv("./sample_data/data/uci-har/test/y_test.txt",sep=r'\s+',header=None,names=["activity_id"])
df_X_test.columns=feature_names
df_y_test_with_actions=df_y_test.merge(df_activities,on="activity_id")

#map activity label with code activity
activity_map = dict(zip(df_activities["activity_id"], df_activities["activity_label"]))
df_y_train=df_y_train['activity_id'].map(activity_map)
df_y_test=df_y_test['activity_id'].map(activity_map)

class validateHumanActivityRecognitionDataSet():

  def __init__(self,dataSetInput):
    self.dataSetInput=dataSetInput


  def validateNulls(self):
    countNan=0

    for itemData in self.dataSetInput:
      has_nan=np.isnan(itemData).any()
      if has_nan:
        countNan=countNan+1


    if  countNan >0 :
     print("Existe presencia de valores nulos en el dataset de entrenamiento")
    else:
     print("No existen valores nulos en la data de entrenamiento")

  def validateBlanks(self):
    countBlank=0

    for itemData in self.dataSetInput:
      has_blank=(itemData=="").any()
      if has_blank:
        countBlank=countBlank+1


    if  countBlank >0 :
     print("Existe presencia de valores blancos en el dataset de entrenamiento")
    else:
     print("No existen valores blancos  en la data de entrenamiento")

  def plotInformation(self,X,y):
    # Aplanar cada muestra: (n_samples, 9, 128) → (n_samples, 1152)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Aplicar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    # Graficar en 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y["activity_id"], cmap='tab10', s=10)
    plt.title("Muestras UCI HAR reducidas con PCA (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(*scatter.legend_elements(), title="Actividad")
    plt.grid(True)
    plt.tight_layout()
    ruta_salida = os.path.join("assets", "cnn", "pca.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')
    plt.show()

  def plotInformation_tsne(self, X, y):
    # Aplanar cada muestra: (n_samples, 9, 128) → (n_samples, 1152)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Aplicar t-SNE para reducir a 2 dimensiones
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate='auto', init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_flat)

    # Graficar en 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y['activity_id'], cmap='tab10', s=10)
    plt.title("Muestras UCI HAR reducidas con t-SNE (2D)")
    plt.xlabel("Dimensión t-SNE 1")
    plt.ylabel("Dimensión t-SNE 2")
    plt.legend(*scatter.legend_elements(), title="Actividad")
    plt.grid(True)
    plt.tight_layout()

    ruta_salida = os.path.join("assets", "cnn", "tsne.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')
    
    plt.show()

  def plotInformation_mejorado(self, X, y):
    activity_labels = {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING"
    }

    # Asegurar que X e y tengan el mismo número de muestras
    n_samples = min(X.shape[0], len(y))
    X = X[:n_samples]
    y = y[:n_samples]

    # Aplanar: (n_samples, 9, 128) → (n_samples, 1152)
    X_flat = X.reshape(n_samples, -1)

    # PCA a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    # Obtener los índices de actividad
    y_labels = y["activity_id"].values if isinstance(y, pd.DataFrame) else y
    y_labels = y_labels.astype(int)

    # Graficar puntos con color por clase
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_labels, cmap='tab10', s=10)

    # Leyenda con nombres reales de las actividades
    handles, _ = scatter.legend_elements()
    unique_ids = sorted(np.unique(y_labels))
    legend_labels = [activity_labels[i] for i in unique_ids]
    plt.legend(handles, legend_labels, title="Actividad", loc="best", fontsize=9)

    # Etiquetas
    plt.title("Muestras UCI HAR reducidas con PCA (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def add_jitter(X_batch, sigma=0.05):
      noise = np.random.normal(loc=0.0, scale=sigma, size=X_batch.shape)
      return X_batch + noise

class ConvertirDataSetHAR(Sequence):
    def __init__(self, X, y, batch_size=64,validation_split=0.0):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples, self.n_channels, self.seq_len = X.shape
        self.validation_split = validation_split
        if isinstance(self.y, pd.DataFrame):
           self.y = self.y.values.flatten()
        elif isinstance(self.y, pd.Series):
           self.y = self.y.values

        #Normalizar los datos , covertir el array 3D de forma (numero_ejemplos, numero_canales, numero_paso_tiempo_por_muestra) a
        #una representacion 2D de forma (numero_ejemplos * numero_canales, numero_paso_tiempo_por_muestra).
        self.X=self.X.reshape(-1,self.seq_len)
        #cada señal de 128 puntos se normaliza a media 0 y desviacion estandar 1
        self.X = (self.X - self.X.mean(axis=1, keepdims=True)) / self.X.std(axis=1, keepdims=True)
        #se retorna a la estructura original del dataset 3D
        self.X = self.X.reshape(self.n_samples, self.n_channels, self.seq_len)

        # Transponer de (n_samples, 9, 128) → (n_samples, 128, 9)
        self.X = self.X.transpose(0, 2, 1)

        if validation_split > 0.0:
            np.random.seed(SEED)
            permuted_indices = np.random.permutation(self.n_samples)
            split_index = int(self.n_samples * (1 - validation_split))
            train_idx, val_idx = permuted_indices[:split_index], permuted_indices[split_index:]
            self.X_train, self.X_val = self.X[train_idx], self.X[val_idx]
            self.y_train, self.y_val = self.y[train_idx], self.y[val_idx]
            self.indexes = np.arange(self.X_train.shape[0])
        else:
           # Índices de los ejemplos
           self.indexes = np.arange(self.X.shape[0])

    def __len__(self):
        # Número total de batches por época
        return int(np.ceil(self.n_samples / self.batch_size))




    def __getitem__(self, idx):
    # Calcular los índices del batch
     batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

     if self.validation_split > 0.0:
        batch_x = self.X_train[batch_indexes]
        y_source = self.y_train
     else:
        batch_x = self.X[batch_indexes]
        y_source = self.y

      # Manejo robusto de las etiquetas
     if isinstance(y_source, pd.DataFrame):
        # Extrae la única columna si es DataFrame
        batch_y = y_source.iloc[batch_indexes].values.flatten()
     elif isinstance(y_source, pd.Series):
        batch_y = y_source.iloc[batch_indexes].values
     else:
        batch_y = y_source[batch_indexes]

     is_training =  self.validation_split == 0.0 or (hasattr(self, "X_train") and np.shares_memory(batch_x, self.X_train))
     if is_training:
      batch_x = add_jitter(batch_x, sigma=0.03)

     return batch_x, batch_y

    def get_validation_data(self):
        # Devuelve los datos de validación si validation_split > 0.0
        if self.validation_split > 0.0:
            return self.X_val, self.y_val
        else:
            return None, None

from tensorflow.keras.regularizers import l2

def cnn_1d_model(input_shape, num_classes):
    # Definir la entrada
    inputs = tf.keras.Input(shape=input_shape)

    # Primera capa convolucional y de pooling
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu", strides=1, padding="same",kernel_regularizer=l2(1e-4))(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Segunda capa convolucional y de pooling
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", strides=1, padding="same",kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Tercera capa convolucional y de pooling
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", strides=1, padding="same",kernel_regularizer=l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Aplanar y añadir Dropout
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.6)(x)

    # Capa de salida
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",kernel_regularizer=l2(1e-4))(x)

    # Crear el modelo usando la API funcional
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn1d_model')

    return model

def validar_balanceo_dataset(y, class_labels=None, title="Distribución de clases en el dataset"):
    plt.figure(figsize=(8, 5))

    # Obtener el conteo de cada clase
    unique, counts = np.unique(y, return_counts=True)

    # Si hay etiquetas de clase proporcionadas
    if class_labels is not None:
        labels = [class_labels[u] for u in unique]
    else:
        labels = unique

    df = pd.DataFrame({"Clase": labels, "Muestras": counts})

    sns.barplot(data=df, x="Clase", y="Muestras", hue="Clase", palette="tab10", legend=False)
    plt.xlabel("Clase")
    plt.ylabel("Número de muestras")
    plt.title("Distribución de clases en el dataset")
    plt.tight_layout()
    plt.show()

signals = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def load_signals(signal_type="train"):
    data = []
    for signal in signals:
        path = f"sample_data/data/uci-har/{signal_type}/Inertial Signals/{signal}_{signal_type}.txt"
        signal_data = np.loadtxt(path)  # shape: (n_samples, 128)
        data.append(signal_data)
    return np.stack(data, axis=1)  # (n_samples, 9, 128)

def load_labels(signal_type="train"):
    path = f"sample_data/data/uci-har/{signal_type}/y_{signal_type}.txt"
    #return np.loadtxt(path).astype(int) - 1  # etiquetas de 0 a 5
    labels = np.loadtxt(path).astype(int) - 1  # de 0 a 5
    return pd.DataFrame(labels, columns=["activity_id"])

X_train=load_signals("train")
y_train = load_labels("train")
objValidation=validateHumanActivityRecognitionDataSet(X_train)
#validacion de nulos en el dataset
objValidation.validateNulls()
#validacion de blancos en el dataset
objValidation.validateBlanks()
#graficar los datos en dos dimensiones
df_y_train_with_actions_cnn1d=y_train.merge(df_activities,on="activity_id")
objValidation.plotInformation(X_train,y_train)
print(y_train)
objValidation.plotInformation_tsne(X_train,y_train)

validar_balanceo_dataset(y_train)

num_activities_cnn1d = y_train["activity_id"].value_counts().sort_index()
ratio_data_set_cnn1d=num_activities_cnn1d.max() / num_activities_cnn1d.min()
if ratio_data_set_cnn1d > 1.5:
  print("El dataset esta desbalanceado")
else :
  print("El dataset NO esta desbalanceado")

X_train = load_signals("train")  # (n_samples, 9, 128)
y_train = load_labels("train")   # (n_samples,)
dataset = ConvertirDataSetHAR(X_train, y_train, batch_size=64,validation_split=0.2)
X_val, y_val = dataset.get_validation_data()
model = cnn_1d_model(input_shape=(128, 9), num_classes=6)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
]

model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

history=  model.fit(
        dataset,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

model.save('model-uci-har.h5')
#Evaluacion del modelo
 # Cargar datos de prueba
X_test = load_signals("test")  # (n_samples, 9, 128)
y_test = load_labels("test")   # (n_samples,)

# Crear generador con la clase personalizada
test_dataset = ConvertirDataSetHAR(X_test, y_test, batch_size=64)

# Evaluar el modelo en el dataset de prueba
results = model.evaluate(test_dataset, verbose=1)

# Realizar predicciones en el conjunto de prueba
#X_test_transposed = X_test.transpose(0, 2, 1)
y_pred = model.predict(test_dataset)

# Convertir las predicciones en etiquetas (la clase con mayor probabilidad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.astype(int)  # Las etiquetas reales del conjunto de prueba

"""**GRAFICA DE VALORES REALES VS PREDICHOS**"""

import matplotlib.pyplot as plt

def plot_loss_historia_keras(history):
    # Graficar el histórico de pérdida durante el entrenamiento
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    ruta_salida = os.path.join("assets", "cnn", "perdida.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')

    plt.show()


def plot_acc_historia_keras(history):
    # Graficar la precisión durante el entrenamiento
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    ruta_salida = os.path.join("assets", "cnn", "precision.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')
    plt.show()

def plot_matriz_confusion(cm):
    # Visualizar la matriz de confusión usando Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Etiqueta predicha')
    plt.ylabel('Etiqueta real')
    plt.title('Matriz de Confusión para el MLP en el dataset MNIST')
    ruta_salida = os.path.join("assets", "cnn", "confusion.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')

def plot_predictions(model):
    X_test = load_signals("test")
    y_test = load_labels("test")
    test_dataset = ConvertirDataSetHAR(X_test, y_test, batch_size=64)

    # Obtener predicciones del modelo
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    # Graficar las predicciones vs. los valores reales
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Real')
    plt.plot(predicted_labels, label='Predicción')
    plt.xlabel('Muestra')
    plt.ylabel('Etiqueta de Actividad')
    plt.title('Predicciones del Modelo vs. Valores Reales')
    plt.legend()
    plt.grid(True)
    ruta_salida = os.path.join("assets", "cnn", "prediccion.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')
    plt.show()

plot_predictions(model)

#graficar Matriz de confunción
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión — Validación')
plt.xlabel('Predicho')
plt.ylabel('Real')
ruta_salida = os.path.join("assets", "cnn", "confusion.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()

print(classification_report(y_true, y_pred_classes))

#graficar perdida durante el entrenamiento
plot_acc_historia_keras(history)

print(f"Precisión final de entrenamiento: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Precisión final de validación: {history.history['val_accuracy'][-1]*100:.2f}%")

plot_loss_historia_keras(history)

print(f"Pérdida final de entrenamiento: {history.history['loss'][-1]*100:.2f}%")
print(f"Pérdida final de validación: {history.history['val_loss'][-1]*100:.2f}%")