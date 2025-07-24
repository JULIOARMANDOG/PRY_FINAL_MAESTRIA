import os
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from tensorflow.keras.models import load_model
import re


def convertir_bits_a_unidades(datos_bits):
    conv_adxl345 = (2 * 16) / (2**13)
    conv_itg3200 = (2 * 2000) / (2**16)
    conv_mma8451q = (2 * 8) / (2**14)

    datos_fisicos = np.empty_like(datos_bits, dtype=np.float32)
    datos_fisicos[:, 0:3] = datos_bits[:, 0:3] * conv_adxl345
    datos_fisicos[:, 3:6] = datos_bits[:, 3:6] * conv_itg3200
    datos_fisicos[:, 6:9] = datos_bits[:, 6:9] * conv_mma8451q

    return datos_fisicos


def crear_ventanas_global(datos, etiqueta, ventana=128, paso=64): 
    X, y = [], []
    for i in range(0, len(datos) - ventana + 1, paso):
        X.append(datos[i:i+ventana])
        y.append(etiqueta)
    return np.array(X), np.array(y)


def procesar_archivo(ruta_archivo, etiqueta):
    datos_limpios = []
    try:
        with open(ruta_archivo, "r") as f:
            lineas = f.readlines()

        for linea in lineas:
            linea = linea.strip().rstrip(';')
            if linea:
                numeros = re.findall(r'-?\d+', linea)
                if len(numeros) == 9:
                    datos_limpios.append([int(n) for n in numeros])

        if not datos_limpios or len(datos_limpios) < 128:
            print(f"Omitido: {ruta_archivo} (datos insuficientes)")
            return np.empty((0, 128, 9)), np.empty((0,))

        datos_bits = np.array(datos_limpios, dtype=np.float32)
        datos_fis = convertir_bits_a_unidades(datos_bits)
        
        X, y = crear_ventanas_global(datos_fis, etiqueta)
        
        print(f"[DEBUG] Ventanas generadas con forma: {X.shape}, dtype: {X.dtype}")
        return X, y

    except Exception as e:
        print(f"Error procesando archivo {ruta_archivo}: {e}")
        return np.array([]), np.array([])


class ImageViewerApp:
    def __init__(self, root, base_folder):
        self.root = root
        self.root.title("Visualizador de Imágenes del Modelo")

        try:
            self.root.state("zoomed")
        except:
            self.root.attributes('-zoomed', True)

        self.base_folder = base_folder
        self.models = {
            "Árbol de Decisión": "arbol",
            "CNN 1D": "cnn",
            "SVM GridSearch": "svm",
            "Prueba concepto": "actividades",
            "Prueba concepto caidas": "caidas"
        }

        self.current_model_folder = os.path.join(self.base_folder, "arbol")
        self.image_files = []
        self.index = 0
        self.last_input = None
        self.graph_canvas = None
        self.prediction_graph_canvas = None 
        self.prediction_toolbar = None 

        self.labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
        self.model = None
        self.scaler = None
        self.model_keras = None
        self.scaler_keras = None 

        try:
            self.model = joblib.load("modelo_svm_gridsearch.pkl")
            self.scaler = joblib.load("scaler_svm.pkl")
        except Exception as e:
            print("⚠️ Error cargando modelo o scaler SVM:", e)

        try:
            self.model_keras = load_model("modelo_transfer_sisfall.h5")
        except Exception as e:
            print("⚠️ Error cargando modelo keras:", e)
        
        # try: 
        #     self.scaler_keras = joblib.load("scaler_svm.pkl") 
        #     print("✅ Scaler para modelo Keras cargado exitosamente.")
        # except Exception as e:
        #     print(f"⚠️ Error cargando scaler Keras (esto es esperado si no tienes uno): {e}")


        main_frame = ttk.Frame(root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.radio_frame = ttk.Frame(main_frame, width=200)
        self.radio_frame.pack(side=LEFT, fill=Y, padx=(0, 10))

        self.selected_model = ttk.StringVar(value="Árbol de Decisión")

        for model_name in self.models.keys():
            rbtn = ttk.Radiobutton(
                self.radio_frame,
                text=model_name,
                value=model_name,
                variable=self.selected_model,
                command=self.change_model,
                bootstyle="info"
            )
            rbtn.pack(anchor="w", pady=5)

        self.viewer_frame = ttk.Frame(main_frame)
        self.viewer_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.label = ttk.Label(self.viewer_frame)
        self.label.pack(padx=10, pady=10, expand=True)

        self.status = ttk.Label(self.viewer_frame, text="", font=("Segoe UI", 10), anchor=CENTER)
        self.status.pack(pady=(0, 5))

        self.btn_frame = ttk.Frame(self.viewer_frame)
        self.btn_frame.pack(pady=10)

        self.prev_button = ttk.Button(self.btn_frame, text="⭨ Anterior", bootstyle=SECONDARY, command=self.prev_image)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.next_button = ttk.Button(self.btn_frame, text="Siguiente ➡", bootstyle=SUCCESS, command=self.next_image)
        self.next_button.grid(row=0, column=1, padx=10)

        self.pred_button = ttk.Button(self.btn_frame, text="Predecir Actividad", bootstyle=PRIMARY, command=self.realizar_prediccion)

        self.load_images()
        self.show_image()

    def load_images(self):
        model_name = self.selected_model.get()
        folder_name = self.models.get(model_name, "arbol")
        self.current_model_folder = os.path.join(self.base_folder, folder_name)

        if not os.path.exists(self.current_model_folder):
            self.image_files = []
            return

        self.image_files = [f for f in os.listdir(self.current_model_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        self.index = 0

    def change_model(self):
        self.load_images()
        self.show_image()

        if self.selected_model.get() in ["Prueba concepto", "Prueba concepto caidas"]:
            self.pred_button.grid(row=0, column=2, padx=10)
        else:
            self.pred_button.grid_forget()

        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None

        
        if self.prediction_graph_canvas:
            self.prediction_graph_canvas.get_tk_widget().destroy()
            self.prediction_graph_canvas = None
        if self.prediction_toolbar: 
            self.prediction_toolbar.destroy()
            self.prediction_toolbar = None

    def show_image(self):
        if not self.image_files:
            self.label.config(image="")
            self.status.config(text="No se encontraron imágenes.")
            return

        image_path = os.path.join(self.current_model_folder, self.image_files[self.index])
        img = Image.open(image_path)
        img = img.resize((1000, 750), Image.Resampling.LANCZOS)
        self.tkimage = ImageTk.PhotoImage(img)

        self.label.config(image=self.tkimage)
        self.status.config(text=f"{self.index + 1} / {len(self.image_files)} - {os.path.basename(image_path)}")

    def next_image(self):
        if not self.image_files:
            return
        self.index = (self.index + 1) % len(self.image_files)
        self.show_image()

    def prev_image(self):
        if not self.image_files:
            return
        self.index = (self.index - 1) % len(self.image_files)
        self.show_image()

    def realizar_prediccion(self):
        modelo_seleccionado = self.selected_model.get()

        if modelo_seleccionado == "Prueba concepto caidas":
            self.prueba_ventana_real_caida()
        else:
            if self.model is None or self.scaler is None:
                self.status.config(text="Modelo o scaler no cargados.")
                return

            try:
                x = np.random.normal(0, 1, (1, 561))
                x_scaled = self.scaler.transform(x)
                y_pred = self.model.predict(x_scaled)[0]
                actividad = self.labels[y_pred - 1]
                self.status.config(text=f"Predicción: {actividad}")

                self.last_input = x[0]
                self.mostrar_grafico("Serie de entrada generada (561 features)")

            except Exception as e:
                self.status.config(text=f"Error en predicción: {e}")

    def mostrar_grafico(self, titulo="Serie de entrada"):
       
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(7, 2.5), dpi=100)
        ax.plot(self.last_input, color="blue")
        ax.set_title(titulo)
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valor")
        ax.grid(True)

        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.viewer_frame)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(before=self.btn_frame, pady=10)

    def prueba_ventana_real_caida(self):
        if self.model_keras is None:
            self.status.config(text="Modelo Keras no cargado.")
            return

       
        if self.prediction_graph_canvas:
            self.prediction_graph_canvas.get_tk_widget().destroy()
            self.prediction_graph_canvas = None
        if self.prediction_toolbar:
            self.prediction_toolbar.destroy()
            self.prediction_toolbar = None

        archivo_prueba = './SisFall/SisFall_dataset/SA01/F05_SA01_R04.txt'
        
        
        X_prueba, y_prueba = procesar_archivo(archivo_prueba, etiqueta=1)

        if X_prueba.shape[0] == 0:
            self.status.config(text="No se pudo cargar el archivo de prueba o tiene datos insuficientes.")
            return

       
        X_prueba_final = X_prueba 
        # if self.scaler_keras:
        #     original_shape = X_prueba.shape
        #     num_ventanas, window_size, num_features = original_shape
        #     X_prueba_reshaped_for_scaler = X_prueba.reshape(-1, num_features)
        #     X_prueba_scaled = self.scaler_keras.transform(X_prueba_reshaped_for_scaler)
        #     X_prueba_final = X_prueba_scaled.reshape(original_shape)
        #     print(f"[DEBUG] Datos escalados para Keras. Nueva forma: {X_prueba_final.shape}")
        # else:
        #     print("[DEBUG] No se aplicó escalado al modelo Keras (scaler_keras no encontrado).")
        

        self.status.config(text=f"Ventanas cargadas: {X_prueba_final.shape[0]}. Prediciendo...")

        predicciones = self.model_keras.predict(X_prueba_final)
        threshold = 0.3
        pred_clases = (predicciones > threshold).astype(int)

        resultados_texto = []
        prediccion_valores = []  
        probabilidad_valores = [] 

        for i, (prob, pred) in enumerate(zip(predicciones, pred_clases)):
            res = "CAÍDA" if pred[0] == 1 else "NO CAÍDA"
            prob_val = prob[0]
            pred_val = pred[0]
            
           # resultados_texto.append(f"Ventana {i+1}: {res} (Prob: {prob_val:.4f})")
            prediccion_valores.append(pred_val)
            probabilidad_valores.append(prob_val)

        
        self.status.config(text="\n".join(resultados_texto[:5]) + ("\n..." if len(resultados_texto) > 5 else ""))
        for r in resultados_texto:
            print(r)

       
        fig_pred, ax_pred = plt.subplots(figsize=(10, 4), dpi=100) 
        
        
        ax_pred.plot(prediccion_valores, 'o-', label='Predicción (0=No Caída, 1=Caída)', markersize=4, linestyle='--', color='red')
        
        
        ax_pred.plot(probabilidad_valores, 'x-', label='Probabilidad de Caída', markersize=3, linestyle='-', color='blue', alpha=0.7)
        
        
        ax_pred.axhline(y=threshold, color='green', linestyle=':', label=f'Umbral ({threshold})')

        ax_pred.set_title(f'Predicciones de Caída para {os.path.basename(archivo_prueba)}')
        ax_pred.set_xlabel('Ventana de Tiempo (Índice)')
        ax_pred.set_ylabel('Predicción / Probabilidad')
        ax_pred.set_yticks([0, 0.5, 1]) 
        ax_pred.set_yticklabels(['NO CAÍDA', '0.5', 'CAÍDA'])
        ax_pred.grid(True)
        ax_pred.legend()
        plt.tight_layout() 

        self.prediction_graph_canvas = FigureCanvasTkAgg(fig_pred, master=self.viewer_frame)
        self.prediction_graph_canvas.draw()
        
       
        self.prediction_graph_canvas.get_tk_widget().pack(before=self.btn_frame, pady=10)

        
        self.prediction_toolbar = NavigationToolbar2Tk(self.prediction_graph_canvas, self.viewer_frame)
        self.prediction_toolbar.update()
        self.prediction_toolbar.pack(before=self.btn_frame, pady=(0,5)) 
        


if __name__ == "__main__":
    app = ttk.Window(themename="darkly")
    folder_resultados = "assets"
    viewer = ImageViewerApp(app, folder_resultados)
    app.mainloop()