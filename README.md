# PRY_FINAL_MAESTRIA
REPOSITORIO PARA PROYECTO FINAL MAESTRIA
COMMIT INICIAL
El proyecto se compone de los siguientes archivos:
- randomForestClassifier.py
- cnn1d.py
- SVM_ProyectoFinal.py
- SVM_Gridsearch_ProyectoFinal.py
- sysfall-transferlearning.py
- resultados.py

Para levantar el ambiente hay que levantar el entorno virtual con el siguiente comando :
 python -m venv tf310_env
 

descargamos el dataser SisFall.zip de la siguiente ruta:
https://drive.google.com/drive/folders/1QBMA4DbZKUzGq1nvWgu9b9r5wIiCo2Lo?usp=sharing

creamos una carpeta denominada SisFall
Descomprimir el contenido del SisFall.zip dentro del directorio SisFall creado en la raiz

luego activamos el entorno virtual
tf310_env\Scripts\activate
 
Una vez levantado el entorno virtual procedemos a instalar las dependencias del entorno
mediante el siguiente comando.

pip install -r requirements.txt

Luego de instalar las librerias dentro del entorno virtual procedemos a ejecutar el archivo resultados.py mediante el siguiente comando:
python resultados.py -> que contiene los resultados de la ejecución de cada modelo utilizado en este proyecto

Para ejecutar cada modelo procedemos de la siguiente forma
- python randomForestClassifier.py
- python cnn1d.py
- python sysfall-transferlearning.py
- python SVM_ProyectoFinal.py
- python SVM_Gridsearch_ProyectoFinal.py

Para cerrar el entorno virtual ejecutamos el comando
deactivate






