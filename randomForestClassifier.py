import requests
import getpass
import zipfile
import io
import os
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


token_pat ="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"

# URL RAW del archivo zip
url_git = "https://raw.githubusercontent.com/JULIOARMANDOG/PRY_FINAL_MAESTRIA/main/uci-har.zip"
path_to_data_uci_har="sample_data/data"
message_download_ok="Archivo ZIP descargado correctamente"
message_donwload_into_path="Archivo extraído en carpeta 'uci-har"
message_download_error=" Error al descargar archivo:"

# Crear cabecera donde se incluya el token PAT para la descarga del archivo zip
headers = {"Authorization": f"token {token_pat}"}
response_git = requests.get(url_git, headers=headers)

if response_git.status_code == 200:
    print(f"✅{message_download_ok}")

    # Extraer zip en memoria y guardar contenido en carpeta 'uci-har'
    with zipfile.ZipFile(io.BytesIO(response_git.content)) as z:
        z.extractall(path_to_data_uci_har)
        print(f"✅ {message_donwload_into_path}")
else:
    print(f"❌ {message_download_error} {response_git.status_code}")
    print(response_git.text)



def mostrar_nube_de_palabras(df_y_train, columna='activity_label'):
    # Contar la frecuencia de cada actividad
    actividades = df_y_train[columna].value_counts().to_dict()

    # Generar el texto repetido según frecuencia
    texto = ' '.join([f"{actividad} " * int(frecuencia) for actividad, frecuencia in actividades.items()])

    # Crear y mostrar la nube
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(texto)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nube de Palabras: Acciones en el dataset de entrenamiento", fontsize=14)
    # Guardar imagen en assets/arbol con nombre nube.png
    ruta_salida = os.path.join("assets", "arbol", "nube.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')

    plt.show()

def identificar_nulos_blancos_vacios(data):
   sumary_validation=pd.DataFrame({
           'Tipo de dato':data.dtypes,
           'Valores NO nulos': data.notnull().sum(),
           'Valores nulos': data.isnull().sum(),
           'Valores blancos': (data == "").sum(),
            'Valores NAN': data.isna().sum().sum(),
           '% Nulos': (data.isnull().sum() / len(data)) * 100,
           'Valores únicos': data.nunique(),
           'Valores duplicados': data.duplicated().sum()
       })
   sumary_validation.index.name="Columna"
   sumary_validation=sumary_validation.sort_values('% Nulos', ascending=False)

   styled_tabla = (
         sumary_validation.style
        .background_gradient(cmap='Reds', subset=['Valores nulos', '% Nulos'])
        .background_gradient(cmap='Blues', subset=['Valores únicos'])
        .set_caption("📊 Resumen de DataFrame estilo info()")
        .hide(axis="index")  # Opcional: oculta índice
    )


   return styled_tabla

def load_data(pathToFile,dataArrayHeader=None):
  if dataArrayHeader is not None :
     return pd.read_csv(pathToFile,sep=r'\s+',header=None)
  else :
      return pd.read_csv(pathToFile,sep=r'\s+',header=None,names=dataArrayHeader)


def graficar_exploracion_inicial(data,title,labely) :
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data.index, y=data.values,hue=data.index, palette="Set2")
    plt.title(f"{title}", fontsize=14)
    plt.ylabel(f"{labely}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Guardar imagen en assets/arbol con nombre nube.png
    ruta_salida = os.path.join("assets", "arbol", "distribucion.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
    plt.savefig(ruta_salida, format='png')
    plt.show()

def plot_nulos(df):
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        fig = px.bar(nulos.sort_values(),
                     title="Cantidad de valores nulos por columna",
                     labels={'value': 'Nulos', 'index': 'Columna'})
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    else:
        print("✅ No hay valores nulos en el DataFrame.")


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

#IDENTIFICAR LA DIMENSION DEL DATASET DE ENTRENAMIENTO
print(df_y_train.isna().sum())

df_X_train.head()

#Validar valores nulos en el dataset de entrenamiento
identificar_nulos_blancos_vacios(df_X_train)

#validar valores nulos en el dataset de variables a predecir
identificar_nulos_blancos_vacios(df_y_train_with_actions)


mostrar_nube_de_palabras(df_y_train_with_actions)

#realizamos un analisis exploratorio de los datos
num_activities = df_y_train_with_actions["activity_label"].value_counts().sort_index()
graficar_exploracion_inicial(num_activities,"Distribución de clases en el conjunto de entrenamiento (UCI HAR)","Número de muestras")

#Vamos a utilizar el ratio entre clases mas comunes y menos comunes para
#identificar si un dataset se encuentra desbalanceado ; el valor que se
#tomara como umbral para determinar desbalanceo sera 1.5 (NO es regla extricta)
#mas bien recomendaciones basadas en heuristica
ratio_data_set=num_activities.max() / num_activities.min()
if ratio_data_set > 1.5:
  print("El dataset esta desbalanceado")
else :
  print("El dataset NO esta desbalanceado")

#Como parte del analisis exploratorio procedemos a visualizar
#como estan distribuidos cada uno de los puntos dentro de cada
#clase a predecir mediante el uso de PCA
#Reducción de dimensionalidad a 2D usando PCA permitiendo bajar de 561 caracteristicas que posee el dataset
#a 2 o 3 componente principales que capturen de mejor forma la variabilidad entre los datos
#permitiendo identificar de mejor forma agrupacion o mezclas entre las diferentes clases
#mostrar el separacion entre las diferentes clases , ya que las identificamos por diferentes colores.
#Detectar outlier o patrones inusuales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_X_train)

# 2. Crear DataFrame para graficar
df_plot = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "activity_label": df_y_train_with_actions["activity_label"].values  # o la columna con clases
})

# 3. Graficar scatter plot coloreado por clase
plt.figure(figsize=(10,7))
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="activity_label", palette="Set2", alpha=0.7)
plt.title("Visualización PCA de muestras por actividad")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# Guardar imagen en assets/arbol con nombre nube.png
ruta_salida = os.path.join("assets", "arbol", "pca.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()


tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(df_X_train)


df_tsne = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "activity_label": df_y_train_with_actions["activity_label"].values
})

# Paso 3: Graficar
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_tsne, x="TSNE1", y="TSNE2", hue="activity_label", palette="Set2", alpha=0.7)
plt.title("Visualización t-SNE de muestras por actividad")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
ruta_salida = os.path.join("assets", "arbol", "tsne.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()


df_X_train.describe().T[['mean', 'std', 'min', 'max']]

df_X_train.duplicated().sum()

scaler=StandardScaler()
scaler.fit(df_X_train)
X_train_scaled=scaler.transform(df_X_train)
X_test_scaled=scaler.transform(df_X_test)

print(df_y_train.isna().sum())

clasification_forest_model=RandomForestClassifier(n_estimators=100,random_state=46)
#train model
clasification_forest_model.fit(X_train_scaled,df_y_train)
#Do prediction
activity_prediction=clasification_forest_model.predict(X_test_scaled)
#print report
print(classification_report(df_y_test,activity_prediction))

importances = clasification_forest_model.feature_importances_
df_importances = pd.DataFrame({
    "feature": df_X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

df_importances["sensor_type"] = df_importances["feature"].str.extract(r'^([tf]Body(?:Acc|Gyro|AccJerk|GyroJerk|BodyAccMag|BodyGyroMag|BodyBodyGyroJerkMag|BodyBodyAccJerkMag|BodyBodyGyroMag|BodyBodyAccMag))')

# Agrupar por tipo de sensor
sensor_importance = df_importances.groupby("sensor_type")["importance"].sum().sort_values(ascending=False)

sensor_importance.plot(kind='bar', figsize=(10, 6), title="Importancia total por tipo de sensor")
plt.ylabel("Importancia acumulada")
plt.tight_layout()
ruta_salida = os.path.join("assets", "arbol", "importancia.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()


# 1. Etiquetas verdaderas (usa la única columna si no tiene nombre)
y_true = df_y_test
labels = sorted(y_true.unique())

cm = confusion_matrix(y_true, activity_prediction, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión - RandomForestClassifier")
plt.tight_layout()
ruta_salida = os.path.join("assets", "arbol", "confusion.png")
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)  # Crear directorio si no existe
plt.savefig(ruta_salida, format='png')
plt.show()