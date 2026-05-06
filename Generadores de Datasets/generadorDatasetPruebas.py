'''
    Autores: Cristian Cabrera y Álex Rafael de la Cruz
    Descripción: Este script genera un dataset de secuencias de palabras de longitud variable a partir de imágenes individuales de letras
    que se encuentran en la carpeta "asl_alphabet_test" (dataset original de kaggle).
    El dataset resultante se guarda en la carpeta "dataset_pruebas", con subcarpetas para cada palabra. 
    Cada imagen de palabra es una secuencia de las letras correspondientes, unidas con una superposición suave para simular una transición natural entre ellas.
    Y son las imagenes usada para hacer las predicciones de las palabras con el modelo entrenado.
'''

import os
import random
import cv2
import numpy as np

print("Preparación del DATASET de secuencias de palabras de longitud variable")

# RUTAS
RUTA_RAIZ = "asl_alphabet_test/asl_alphabet_test/"
RUTA_DESTINO = "dataset_pruebas"

# DICCIONARIO DE PALABRAS DE 3, 4 Y 5 LETRAS
PALABRAS = [
    'ALTA', 'BESO', 'WIFI', 'FOTO', 'ISLA', 'QUIS','MOTO', 'LUNA', 
    'SOL', 'PAN', 'LUZ', 'OLA', 'UVA', 'QUE', 'PEZ', 'LOS' , 
    'COCHE', 'ARBOL', 'PEDRO', 'PERRO', 'SILLA', 'LIBRO', 'AVION'
]

# PARÁMETROS
MUESTRAS_POR_PALABRA = 1
TAM_LETRA = 128
OVERLAP = 5 # Superposición en píxeles

# FUNCIÓN PARA LA MEZCLA DE IMÁGENES CON SUPERPOSICIÓN
def mezclar_imagenes(img1, img2, overlap):
    h, w1 = img1.shape[:2]
    _, w2 = img2.shape[:2]
    
    # Crear el lienzo para la mezcla
    ancho_total = w1 + w2 - overlap
    lienzo = np.zeros((h, ancho_total, 3), dtype=np.uint8)
    
    # Colocar la primera imagen en el lienzo
    lienzo[:, :w1] = img1
    
    # Crear una máscara de gradiente para la zona de overlap (de 1 a 0)
    mask = np.linspace(1, 0, overlap).reshape(1, -1, 1).repeat(h, axis=0).repeat(3, axis=2)
    mask_inv = 1 - mask
    
    # Zona de mezcla
    zona_overlap_1 = img1[:, w1 - overlap:]
    zona_overlap_2 = img2[:, :overlap]
    
    mezcla = zona_overlap_1 * mask + zona_overlap_2 * mask_inv
    mezcla = mezcla.astype(np.uint8)
    
    # Colocar la segunda imagen en el lienzo
    lienzo[:, w1 - overlap:w1] = mezcla
    lienzo[:, w1:] = img2[:, overlap:]
    
    return lienzo

# --- CORRECCIÓN 1: BUCLE DINÁMICO ---
def generar_palabra(letras_imgs, overlap):
    # Cogemos la primera letra como base
    res = letras_imgs[0]
    # Iteramos desde la segunda letra hasta el final, uniéndolas una a una
    for i in range(1, len(letras_imgs)):
        res = mezclar_imagenes(res, letras_imgs[i], overlap)
    return res


# Proceso de creación del dataset
if not os.path.exists(RUTA_DESTINO):
    os.makedirs(RUTA_DESTINO)

for palabra in PALABRAS:
    print(f"\nGenerando muestras para la palabra: {palabra} ({len(palabra)} letras)")
    path_palabra = os.path.join(RUTA_DESTINO, palabra)
    os.makedirs(path_palabra, exist_ok=True)

    for i in range(MUESTRAS_POR_PALABRA):
        letras_imgs = []
        for letra in palabra:
            nombre_archivo = f"{letra}_test.jpg"
            ruta_imagen = os.path.join(RUTA_RAIZ, nombre_archivo)
            
            img = cv2.imread(ruta_imagen)
            
            if img is None:
                print(f"Error: OpenCV no encontró o no pudo leer la imagen en {ruta_imagen}")
                continue 
                
            # Procesamiento
            img = img[5:-5, 5:-5] 
            img = cv2.resize(img, (TAM_LETRA, TAM_LETRA))
            letras_imgs.append(img)
            
        # --- CORRECCIÓN 2: VALIDACIÓN DINÁMICA ---
        # Comprobamos que hemos cargado tantas imágenes como letras tiene la palabra
        if len(letras_imgs) == len(palabra):
            secuencia = generar_palabra(letras_imgs, OVERLAP)
            nombre_img = f"{palabra}_{i:03d}.jpg"
            cv2.imwrite(os.path.join(path_palabra, nombre_img), secuencia)
            print(f"-> Guardada imagen de {secuencia.shape[1]} píxeles de ancho.")
        else:
            print(f"Omitiendo la palabra '{palabra}' porque faltan letras.")

print(f"\nDataset creado con éxito en la carpeta '{RUTA_DESTINO}'")