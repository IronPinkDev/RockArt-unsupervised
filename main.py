import cv2
import os
import pandas as pd
import numpy as np

# Ruta de la carpeta con las imágenes
carpeta = "Data/Aumentados_iguales"

# Inicializar el descriptor HOG de OpenCV
hog = cv2.HOGDescriptor()

# Listas para almacenar los resultados
hog_features_list = []
filenames = []

for archivo in os.listdir(carpeta):
    ruta_imagen = os.path.join(carpeta, archivo)
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Redimensionar la imagen al tamaño esperado por defecto por HOGDescriptor (64x128)
        img_resized = cv2.resize(img, (64, 128))
        features = hog.compute(img_resized)
        hog_features_list.append(features.flatten())
        filenames.append(archivo)

# Crear un DataFrame de pandas
df_hog_cv = pd.DataFrame(hog_features_list)
df_hog_cv.insert(0, 'filename', filenames)

# Mostrar el DataFrame
print(df_hog_cv)