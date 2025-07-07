# Usa una imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir dash \
    dash-bootstrap-components \
    plotly \
    pandas \
    scikit-learn

# Expone el puerto por defecto de Dash
EXPOSE 8050

# Ejecuta la aplicación
CMD ["python", "app.py"]
