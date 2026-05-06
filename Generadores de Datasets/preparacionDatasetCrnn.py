'''
  Autores: Cristian Cabrera y Álex Rafael de la Cruz
  Descripción:
  Este código prepara un dataset de secuencias de palabras de 3, 4 y 5 letras para entrenamiento de una CRNN.
  Toma imágenes de letras individuales del dataset ASL Alphabet, las mezcla con una superposición
  y las utiliza para crear secuencias de palabras, pegándolas en un lienzo universal de 620x128 (con padding negro a la derecha).
  Guarda las imágenes resultantes en una nueva carpeta organizada por longitud de palabra.

'''


import os
import random
import cv2
import numpy as np

print("Generador de Dataset CRNN (Lienzo Universal 620x128 con selección aleatoria)")

# Cambia esto a la ruta exacta donde está la carpeta que contiene las carpetas A, B, C...
RUTA_RAIZ = "asl_alphabet_train/asl_alphabet_train/" 
RUTA_DESTINO = "dataset_robusto_crnn"
LETRAS_ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

TAM_LETRA = 128
OVERLAP = 5
ANCHO_LIENZO = 620  # El tamaño máximo exacto para 5 letras

# Cantidad de imágenes a generar para forzar el aprendizaje
MUESTRAS_POR_LONGITUD = 3000 

def mezclar_imagenes(img1, img2, overlap):
    h, w1 = img1.shape[:2]
    _, w2 = img2.shape[:2]
    ancho_total = w1 + w2 - overlap
    lienzo = np.zeros((h, ancho_total, 3), dtype=np.uint8)
    lienzo[:, :w1] = img1
    
    mask = np.linspace(1, 0, overlap).reshape(1, -1, 1).repeat(h, axis=0).repeat(3, axis=2)
    mask_inv = 1 - mask
    
    zona_overlap_1 = img1[:, w1 - overlap:]
    zona_overlap_2 = img2[:, :overlap]
    mezcla = (zona_overlap_1 * mask + zona_overlap_2 * mask_inv).astype(np.uint8)
    
    lienzo[:, w1 - overlap:w1] = mezcla
    lienzo[:, w1:] = img2[:, overlap:]
    return lienzo

os.makedirs(RUTA_DESTINO, exist_ok=True)

# Bucle principal de generación
for longitud in [3, 4, 5]:
    print(f"\nGenerando {MUESTRAS_POR_LONGITUD} secuencias de {longitud} letras...")
    
    for i in range(MUESTRAS_POR_LONGITUD):
        # Crear una palabra aleatoria (ej: "XQA", "BMRT")
        palabra_aleatoria = "".join(random.choice(LETRAS_ALFABETO) for _ in range(longitud))
        
        # Cargar las imágenes entrando en las subcarpetas
        letras_imgs = []
        valido = True
        
        for letra in palabra_aleatoria:
            carpeta_letra = os.path.join(RUTA_RAIZ, letra)
            
            # Comprobamos si la carpeta existe
            if not os.path.exists(carpeta_letra):
                print(f"  [!] No encuentro la carpeta: {carpeta_letra}")
                valido = False
                break
                
            # Listamos todas las imágenes de esa letra y elegimos una al azar
            archivos_letra = [f for f in os.listdir(carpeta_letra) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not archivos_letra:
                print(f"  [!] La carpeta {carpeta_letra} está vacía.")
                valido = False
                break
                
            archivo_elegido = random.choice(archivos_letra)
            ruta_imagen = os.path.join(carpeta_letra, archivo_elegido)
            
            # Leemos y procesamos
            img = cv2.imread(ruta_imagen)
            if img is None:
                print(f"  [!] Error leyendo {ruta_imagen}")
                valido = False
                break
                
            img = img[5:-5, 5:-5] # Recorte de bordes
            img = cv2.resize(img, (TAM_LETRA, TAM_LETRA))
            letras_imgs.append(img)
            
        if not valido: 
            continue # Si falló alguna letra, saltamos a la siguiente secuencia
        
        # Unir las letras con overlap
        secuencia = letras_imgs[0]
        for j in range(1, len(letras_imgs)):
            secuencia = mezclar_imagenes(secuencia, letras_imgs[j], OVERLAP)
            
        # Pegar en el lienzo universal de 620x128 (Padding negro a la derecha)
        lienzo_final = np.zeros((TAM_LETRA, ANCHO_LIENZO, 3), dtype=np.uint8)
        w_secuencia = secuencia.shape[1]
        lienzo_final[:, :w_secuencia] = secuencia
        
        # Guardar (Formato: ABCD_001.jpg)
        nombre_archivo = f"{palabra_aleatoria}_{i:04d}.jpg"
        cv2.imwrite(os.path.join(RUTA_DESTINO, nombre_archivo), lienzo_final)

print(f"\n¡Dataset creado!")
