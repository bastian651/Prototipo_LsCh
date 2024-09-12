import os
import cv2

# Este script contine  variables que se llegan a utilizar para ejecutar el main.py

# Rutas
ROOT_PATH = r"C:\Users\Usuario\Desktop\Entornos virtuales\Prototipo_LsCh\Prototipo_v2"
DATA_PATH = os.path.join(ROOT_PATH, "data2")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

# Variables correspondientes a la cantidad minima y máxima de frames y la cantidad de puntos clave
MAX_LENGTH_FRAMES = 15
LENGTH_KEYPOINTS = 1662
MIN_LENGTH_FRAMES = 5

# Nombre del modelo
MODEL_NAME = 'lsch_24.h5'
