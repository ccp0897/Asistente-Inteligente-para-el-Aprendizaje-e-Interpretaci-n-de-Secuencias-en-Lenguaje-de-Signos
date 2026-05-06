# Asistente Inteligente para el Aprendizaje e Interpretación de Secuencias en Lenguaje de Signose-Signos
Trabajo Final - PIA

## Autores
Cristian Cabrera y Álex Rafael de la Cruz


## 📂 Estructura y Contenido de la Entrega

La entrega se organiza en los siguientes directorios y archivos principales:

### 1. Análisis de Datos
* **`EDA/`**: Carpeta que contiene los cuadernos y análisis exploratorios realizados sobre los conjuntos de datos. Incluye el balanceo de clases, frecuencias de letras, distribuciones de píxeles y validación de las superposiciones y el *padding* (lienzo de 620x128px).

### 2. Generación de Datos Sintéticos
* **`Generadores de Datasets/`**: Directorio que contiene los 3 scripts de Python utilizados para la creación programática de las imágenes a partir del alfabeto base:
  * Script para la generación del dataset de entrenamiento de la **CNN** (palabras de longitud fija de 4 letras).
  * Script para la generación del dataset de entrenamiento de la **CRNN** (secuencias aleatorias de longitud variable: 3, 4 y 5 letras sobre lienzo negro).
  * Script para la generación de secuencias de **pruebas/test** utilizadas en la inferencia en frío.

### 3. Modelado y Entrenamiento (Notebooks)
* **`CNN_multihead_Signos.ipynb`**: Cuaderno interactivo que detalla la arquitectura, el entrenamiento  y la evaluación del clasificador estático de salidas múltiples.
* **`CRNN_signos.ipynb`**: Cuaderno interactivo con la implementación del modelo dinámico de secuencias de signos.

### 4. Modelos Entrenados (Archivos H5)
* **`modelo_vgg16_v2.h5`**: Pesos optimizados del modelo estático.
* **`modelo_crnn_v2.h5`**: Pesos optimizados y cabeza de inferencia extraída del modelo secuencial, listos para integrarse en producción.

### 5. Despliegue y Aplicación Web
* **`app.py`**: Servidor *backend* programado en Flask. Se encarga de recibir la imagen del usuario, preprocesarla, inyectarla en el modelo y devolver la traducción.
* **`index.html`**: Interfaz visual interactiva (*frontend*) para que el usuario pueda interactuar con el modelo de IA.
* **`requirements.txt`**: Listado exacto de las librerías y versiones de Python necesarias para replicar el entorno de ejecución del proyecto.