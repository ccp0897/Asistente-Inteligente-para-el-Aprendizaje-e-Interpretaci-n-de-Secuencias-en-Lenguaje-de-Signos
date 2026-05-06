'''
  Autores: Cristian Cabrera y Álex Rafael de la Cruz
  Descripción: Este código prepara un dataset de secuencias de palabras de 4 letras para entrenamiento de una CNN.
  Toma imágenes de letras individuales del dataset ASL Alphabet, las mezcla con una superposición
  y las utiliza para crear secuencias de palabras, y guarda las imágenes resultantes en una nueva carpeta organizada por palabra.
'''


import os
import random
import cv2
import numpy as np
print("Preparación del DATASET de secuencias de palabras de 4 letras")


# RUTAS

RUTA_RAIZ = "asl_alphabet_train/asl_alphabet_train"
RUTA_DESTINO = "dataset_4_letras"

# DICCIONARIO DE PALABRAS DE 4 LETRAS

PALABRAS = [
    'AMOR', 'BALA', 'BESO', 'CABO', 'CENA', 'DADO', 'DEDO', 'EDAD', 'ERRE', 'FASE',
    'FOTO', 'GATO', 'GOTA', 'HIJO', 'HOLA', 'ISLA', 'IRSE', 'JEFE', 'JOTA', 'KILO',
    'LADO', 'LORO', 'MANO', 'MESA', 'NAVE', 'NIDO', 'OIDO', 'OCHO', 'PAPA', 'PELO',
    'QUIS', 'RAMA', 'RISA', 'SAPO', 'SOLA', 'TEMA', 'UNIR', 'UVAS', 'VINO', 'VIDA',
    'WIFI', 'YATE', 'YESO', 'ZUMO', 'ZONA', 'ALTA', 'BAJO', 'CINE', 'DUDA', 'TAZA'
]

# PARÁMETROS DE CREACIÓN DEL DATASET (MUESTRAS POR PALABRA, TAMAÑO DE LA IMAGEN, Y SUPERPOSICIÓN DE LAS IMAGENES)

MUESTRAS_POR_PALABRA = 400
TAM_LETRA = 128
OVERLAP = 5 # Superposición en píxeles

# FUNCIÓN PARA LA MEZCLA DE IMÁGENES CON SUPERPOSICIÓN, MEZCLA LADO DERECHO DE IMG1 CON EL LADO IZQUIERDO DE IMG2
def mezclar_imagenes(img1, img2, overlap):
    h, w1 = img1.shape[:2]
    _, w2 = img2.shape[:2]
    
    # Crear el lienzo para la mezcla
    ancho_total = w1 + w2 - overlap
    lienzo = np.zeros((h, ancho_total, 3), dtype=np.uint8)
    
    # Colocar la primera imagen en el lienzo
    lienzo[:, :w1] = img1
    
    # Crear una máscara de gradiente para la zona de overlap (de 1 a 0)
    # y otra inversa (de 0 a 1)
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

# FUNCIÓN PARA GENERAR UNA SECUENCIA DE IMÁGENES PARA UNA PALABRA DE 4 LETRAS
def generar_palabra(letras_imgs, overlap):
    res = mezclar_imagenes(letras_imgs[0], letras_imgs[1], overlap)
    res = mezclar_imagenes(res, letras_imgs[2], overlap)
    res = mezclar_imagenes(res, letras_imgs[3], overlap)
    return res


# Proceso de creción del dataset

if not os.path.exists(RUTA_DESTINO):
    os.makedirs(RUTA_DESTINO)

for palabra in PALABRAS:
    path_palabra = os.path.join(RUTA_DESTINO, palabra)
    os.makedirs(path_palabra, exist_ok=True)

    for i in range(MUESTRAS_POR_PALABRA):
        letras_imgs = []
        for letra in palabra:
            folder_letra = os.path.join(RUTA_RAIZ, letra)
            archivo = random.choice(os.listdir(folder_letra))
            img = cv2.imread(os.path.join(folder_letra, archivo))
            img = img[5:-5, 5:-5] # Recortar bordes para eliminar ruido
            img = cv2.resize(img, (TAM_LETRA, TAM_LETRA))
            letras_imgs.append(img)
        secuencia = generar_palabra(letras_imgs, OVERLAP)
        
        nombre_img = f"{palabra}_{i:03d}.jpg"
        cv2.imwrite(os.path.join(path_palabra, nombre_img), secuencia)

print("Dataset creado con éxito en la carpeta 'dataset_4_letras'")
