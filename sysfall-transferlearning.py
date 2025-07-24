import numpy as np
import os
import re
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Ruta del dataset
ruta_dataset = './SisFallDataSet/'

def convertir_bits_a_unidades(datos_bits):
    conv_adxl345 = (2 * 16) / (2**13)
    conv_itg3200 = (2 * 2000) / (2**16)
    conv_mma8451q = (2 * 8) / (2**14)

    datos_fisicos = np.empty_like(datos_bits, dtype=np.float32)
    datos_fisicos[:, 0:3] = datos_bits[:, 0:3] * conv_adxl345
    datos_fisicos[:, 3:6] = datos_bits[:, 3:6] * conv_itg3200
    datos_fisicos[:, 6:9] = datos_bits[:, 6:9] * conv_mma8451q

    return datos_fisicos

def crear_ventanas(datos, etiqueta, ventana=128, paso=64):
    X, y = [], []
    for i in range(0, len(datos) - ventana + 1, paso):
        X.append(datos[i:i+ventana])
        y.append(etiqueta)
    return np.array(X), np.array(y)

def procesar_archivo(ruta_archivo, etiqueta):
    try:
        with open(ruta_archivo, 'r') as f:
            lineas = f.readlines()

        datos_limpios = []
        for linea in lineas:
            numeros = re.findall(r'-?\d+', linea)
            if len(numeros) == 9:
                datos_limpios.append([int(n) for n in numeros])

        if not datos_limpios or len(datos_limpios) < 128:
            print(f"Omitido: {ruta_archivo}")
            return np.empty((0, 128, 9)), np.empty((0,))

        datos_bits = np.array(datos_limpios, dtype=np.float32)
        datos_fis = convertir_bits_a_unidades(datos_bits)
        return crear_ventanas(datos_fis, etiqueta)

    except Exception as e:
        print(f"Error procesando {ruta_archivo}: {e}")
        return np.empty((0, 128, 9)), np.empty((0,))

# Cargar archivos y etiquetas
archivos, etiquetas = [], []
for carpeta in os.listdir(ruta_dataset):
    if carpeta.startswith(('SA', 'SE')):
        ruta_carpeta = os.path.join(ruta_dataset, carpeta)
        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith('.txt'):
                etiqueta = 1 if archivo.startswith('F') else 0
                archivos.append(os.path.join(ruta_carpeta, archivo))
                etiquetas.append(etiqueta)

# Procesar los archivos
X_total, y_total = [], []
for ruta_archivo, etiqueta in zip(archivos, etiquetas):
    X_vent, y_vent = procesar_archivo(ruta_archivo, etiqueta)
    if len(X_vent):
        X_total.append(X_vent)
        y_total.append(y_vent)

X_total = np.concatenate(X_total, axis=0)
y_total = np.concatenate(y_total, axis=0)
print(f"✅ Datos: {X_total.shape}, Etiquetas: {y_total.shape}")

# -------------------
# Transfer Learning
# -------------------
modelo_preentrenado = load_model('model-uci-har.h5')

# Fine-tuning parcial: descongelar desde la capa 5
for capa in modelo_preentrenado.layers[:5]:
    capa.trainable = False
for capa in modelo_preentrenado.layers[5:]:
    capa.trainable = True

# Redefinir la cabeza del modelo
x = modelo_preentrenado.layers[-2].output
x = layers.Dense(64, activation='relu', name='tl_dense')(x)
x = layers.Dropout(0.5, name='tl_dropout')(x)
salida = layers.Dense(1, activation='sigmoid', name='tl_output')(x)

modelo_transfer = models.Model(inputs=modelo_preentrenado.input, outputs=salida)

modelo_transfer.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

# Separar entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_total, y_total, test_size=0.2, random_state=46)

# Calcular pesos para clases desbalanceadas
pesos = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
pesos_dict = dict(enumerate(pesos))
print(f"Pesos por clase: {pesos_dict}")

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
modelo_transfer.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    class_weight=pesos_dict,
                    callbacks=[early_stop])

# Evaluación
loss, acc = modelo_transfer.evaluate(X_val, y_val)
print(f"\n🎯 Accuracy en validación: {acc:.4f}")

# Reporte detallado
y_pred = (modelo_transfer.predict(X_val) > 0.5).astype(int)
print("\n📊 Reporte de clasificación:")
print(classification_report(y_val, y_pred, digits=4))

modelo_transfer.save('modelo_transfer_sisfall.h5')
print("💾 Modelo guardado como 'modelo_transfer_sisfall.h5'")