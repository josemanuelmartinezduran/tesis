import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import midi_processor
import os

# --- Configuración ---
SEQ_LENGTH = 4      # Cuántas notas mira hacia atrás para predecir la siguiente
NUM_FEATURES = 4    # [Nota, Duración, Grado, Intervalo]
EPOCHS = 100        # Iteraciones de entrenamiento
BATCH_SIZE = 16

def generate_dummy_data(num_samples=50):
    """
    Genera datos sintéticos basados en la escala de prueba para que el modelo tenga algo que aprender.
    En un caso real, esto iteraría sobre una carpeta de archivos .mid
    """
    print("Generando datos de entrenamiento sintéticos...")
    data_matrices = []
    
    # Creamos variaciones de la escala C Major ascendente y descendente
    base_file = 'temp_train.mid'
    
    # Generamos una matriz base usando el procesador
    midi_processor.create_test_midi(base_file)
    base_matrix = midi_processor.midi_to_matrix(base_file)
    
    # Replicamos y variamos ligeramente para tener "datos"
    for _ in range(num_samples):
        # Añadir ruido aleatorio muy leve a la duración o intervalos
        noisy_matrix = base_matrix.copy()
        # Variación simple: a veces repetimos notas
        if np.random.rand() > 0.5:
            noisy_matrix = np.vstack([noisy_matrix, noisy_matrix])
        
        data_matrices.append(noisy_matrix)
        
    if os.path.exists(base_file):
        os.remove(base_file)
        
    return data_matrices

def prepare_sequences(matrices, seq_length):
    X = []
    y = []
    
    for matrix in matrices:
        if len(matrix) <= seq_length:
            continue
            
        for i in range(len(matrix) - seq_length):
            X.append(matrix[i:i+seq_length])
            y.append(matrix[i+seq_length])
            
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='relu', return_sequences=False),
        Dense(32, activation='relu'),
        Dense(NUM_FEATURES) # Salida: 4 valores continuos (Regresión)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_melody(model, seed_sequence, length=20):
    """Genera una nueva melodía a partir de una semilla."""
    generated = list(seed_sequence)
    current_seq = np.array(seed_sequence) # Shape (seq_length, 4)
    
    print("Generando nueva melodía...")
    
    for _ in range(length):
        # Predecir siguiente paso
        # Reshape a (1, seq_length, 4)
        input_seq = current_seq.reshape(1, SEQ_LENGTH, NUM_FEATURES)
        prediction = model.predict(input_seq, verbose=0)[0]
        
        # Post-procesamiento básico para limpiar la salida (opcional)
        # Por ejemplo, redondear la nota a entero, o la duración a 0.25
        # Aquí lo dejamos crudo para ver qué "piensa" la red, 
        # pero el convertidor a MIDI ya redondea el intervalo.
        
        generated.append(prediction)
        
        # Actualizar secuencia (deslizar ventana)
        # Quitamos el primero, añadimos la predicción al final
        current_seq = np.vstack([current_seq[1:], prediction])
        
    return np.array(generated)

if __name__ == "__main__":
    # 1. Obtener datos
    # En producción: Cargarías tus archivos MIDI reales aquí
    raw_matrices = generate_dummy_data(num_samples=100)
    
    # 2. Preparar secuencias
    X, y = prepare_sequences(raw_matrices, SEQ_LENGTH)
    
    print(f"Dimensiones de entrenamiento: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("No hay suficientes datos para entrenar.")
        exit()
        
    # 3. Construir y Entrenar
    model = build_model((SEQ_LENGTH, NUM_FEATURES))
    model.summary()
    
    print(f"Entrenando por {EPOCHS} épocas...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 4. Generar
    # Usamos la primera secuencia de los datos como semilla
    seed = X[0]
    new_melody_matrix = generate_melody(model, seed, length=16)
    
    print("\nMatriz Generada (primeras 5 filas):")
    print(new_melody_matrix[:5])
    
    # 5. Guardar MIDI
    output_file = "melodia_generada_ia.mid"
    midi_processor.matrix_to_midi(new_melody_matrix, output_file, start_note=60)
    print(f"\n¡Listo! Melodía guardada en '{output_file}'.")
