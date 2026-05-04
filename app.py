import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM
from tensorflow.keras.models import Model

app = Flask(__name__)
CORS(app)

ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PALABRAS = [
    'AMOR', 'BALA', 'BESO', 'CABO', 'CENA', 'DADO', 'DEDO', 'EDAD', 'ERRE', 'FASE',
    'FOTO', 'GATO', 'GOTA', 'HIJO', 'HOLA', 'ISLA', 'IRSE', 'JEFE', 'JOTA', 'KILO',
    'LADO', 'LORO', 'MANO', 'MESA', 'NAVE', 'NIDO', 'OIDO', 'OCHO', 'PAPA', 'PELO',
    'QUIS', 'RAMA', 'RISA', 'SAPO', 'SOLA', 'TEMA', 'UNIR', 'UVAS', 'VINO', 'VIDA',
    'WIFI', 'YATE', 'YESO', 'ZUMO', 'ZONA', 'ALTA', 'BAJO', 'CINE', 'DUDA', 'TAZA'
]


# ARQUITECTURA PARA MODELO 2 (CRNN)
def build_modelo_crnn():
    entrada_imagen = Input(shape=(620, 128, 1), name='imagen_entrada')
    
    # Bloque CNN
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(entrada_imagen)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # Reajuste para la parte Recurrente (RNN)
    x = Reshape(target_shape=(155, 2048))(x)
    x = Dense(256, activation='relu')(x)

    # Capas Bidireccionales LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Capa de salida
    salida_prediccion = Dense(len(ALFABETO) + 1, activation='softmax', name='prediccion_softmax')(x)
    
    return Model(inputs=entrada_imagen, outputs=salida_prediccion)

# CARGAMOS LOS DOS MODELOS
try:
# Carga Modelo 1 (VGG16)
    model1 = tf.keras.models.load_model('modelo_vgg16.h5', compile=False)
    
    # Carga Modelo 2 (CRNN)
    model2 = build_modelo_crnn()
    model2.load_weights('modelo_crnn.h5')
    print("Modelos 1 y 2 cargados con éxito")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")

# Preparacion de imagen
def prepare_image(image_bytes, version):
    img = Image.open(io.BytesIO(image_bytes))
    
    if version == "1":
        # Modelo 1: RGB, 497x128
        img = img.convert('RGB').resize((497, 128))
        img_array = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    else:
        # Modelo 2: Grayscale, 620x128, Transpuesta (.T)
        img = img.convert('L').resize((620, 128))
        img_norm = np.array(img).astype(np.float32) / 255.0
        img_batch = np.expand_dims(np.expand_dims(img_norm.T, axis=-1), axis=0)
        return img_batch

# Recoleccion de imagenes
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    version = request.form.get('version')# Agregamos una version para diferenciar entre los modelos

    if not file:
        return jsonify({"error": "No hay imagen"}), 400

    try:
        img_bytes = file.read()
        # Verificamos que en la version 1 sea con un ancho y alto concreto
        if version == "1":
            img_raw = Image.open(io.BytesIO(img_bytes))
            ancho, alto = img_raw.size
            if ancho != 497 or alto != 128:
                return jsonify({
                    "status": "error",
                    "error": "Imagen no compatible con la Versión 1"
                }), 400
        input_data = prepare_image(img_bytes, version)
        
        res_texto = ""

        # Lógica Modelo 1
        if version == "1":
            prediction = model1.predict(input_data)
            # Recolectamos las letras del array
            for i in range(len(prediction)):
                indice_letra = np.argmax(prediction[i])
                if indice_letra < len(ALFABETO):
                    res_texto += ALFABETO[indice_letra]
                else:
                    res_texto += "?" # Por si predice algo fuera del alfabeto
            # Validacion del diccionario (Solo en el Modelo 1)
            if res_texto not in PALABRAS:
                return jsonify({
                    "status": "warning",
                    "resultado": f"Detectado: {res_texto}",
                    "error": "El modelo no ha entrenado con esa imagen"
                })

        
        else:
            # Lógica Modelo 2
            pred = model2.predict(input_data, verbose=0)
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            
            # Decodificación CTC
            decoded = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
            
            # El resultado viene en una matriz de índices de Keras
            indices = decoded[0].numpy() if hasattr(decoded[0], "numpy") else decoded[0]
            for num in indices:
                if num != -1 and num < len(ALFABETO):
                    res_texto += ALFABETO[int(num)]
        
        return jsonify({
            "status": "success",
            "resultado": f"Resultado (Modelo v{version}): {res_texto if res_texto else 'No se detectaron letras'}"
        })

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)