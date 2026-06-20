import numpy as np
import tensorflow as tf
import procesador_midi as midi_processor
import os
import constructor_red as network_builder
import mido
import filtro_arbol_decision as decision_tree_filter

# --- Configuración ---
SEQ_LENGTH = 4      # Cuántas notas mira hacia atrás para predecir la siguiente
NUM_FEATURES = 4    # [Nota, Duración, Grado, Intervalo]
EPOCHS = 100        # Iteraciones de entrenamiento
BATCH_SIZE = 16

def load_midi_dataset(folder_path):
    """
    Carga todos los archivos MIDI (.mid o .midi) de una carpeta y los convierte
    a matrices numéricas utilizando midi_processor.
    """
    print(f"Cargando dataset desde la carpeta: {folder_path}...")
    data_matrices = []
    
    if not os.path.exists(folder_path):
        print(f"Advertencia: La carpeta '{folder_path}' no existe. Creándola automáticamente.")
        os.makedirs(folder_path, exist_ok=True)
        return data_matrices

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mid', '.midi'))]
    if len(files) == 0:
        print(f"Advertencia: No se encontraron archivos MIDI en '{folder_path}'.")
        return data_matrices

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            matrix = midi_processor.midi_to_matrix(filepath)
            if len(matrix) > 0:
                data_matrices.append(matrix)
                print(f"  Procesado exitosamente: {filename} (forma: {matrix.shape})")
        except Exception as e:
            print(f"  Error al procesar '{filename}': {e}")
            
    return data_matrices

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

def build_model(input_shape, structure=None):
    """
    Construye el modelo llamando al generador dinámico de capas.
    Si no se proporciona una estructura, intenta cargarla desde arquitectura_tensor.txt.
    Si no existe, se usa una arquitectura por defecto.
    """
    if structure is None:
        default_file = os.path.join(os.path.dirname(__file__), 'arquitectura_tensor.txt')
        alt_file = os.path.join(os.path.dirname(__file__), 'arquitactura_tensor.txt')
        target_file = default_file if os.path.exists(default_file) else (alt_file if os.path.exists(alt_file) else None)
        
        if target_file:
            print(f"Cargando arquitectura predeterminada desde el archivo matricial: {target_file}")
            structure = network_builder.load_architecture_from_file(target_file)
        else:
            # Estructura por defecto que equivale a la red original:
            # LSTM de 64 neuronas seguido de una capa Dense de 32 neuronas
            structure = [
                (network_builder.LAYER_LSTM, 64, 'relu'),
                (network_builder.LAYER_DENSE, 32, 'relu')
            ]
    return network_builder.build_dynamic_model(input_shape, structure, output_dim=NUM_FEATURES)

def generate_melody(model, seed_sequence, length=20):
    """Genera una nueva melodía a partir de una semilla."""
    generated = list(seed_sequence)
    current_seq = np.array(seed_sequence) # Forma: (seq_length, 4)
    
    print("Generando nueva melodía...")
    
    for _ in range(length):
        # Predecir siguiente paso
        # Ajustar la forma (reshape) a (1, seq_length, 4)
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

def create_fallback_midi(filepath):
    """
    Crea un archivo MIDI de prueba más largo para asegurar que haya suficientes secuencias para entrenar.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    sixteenth = int(ticks_per_beat / 4)
    
    # Escala de Do Mayor ascendente y descendente repetida
    scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62]
    notes = []
    for _ in range(5):
        for note in scale:
            notes.append((note, sixteenth * (2 if np.random.rand() > 0.5 else 1)))
            
    for pitch, duration in notes:
        track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=64, time=duration))
        
    mid.save(filepath)
    print(f"Creado archivo MIDI de prueba largo en: {filepath}")

if __name__ == "__main__":
    # 1. Obtener datos
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    # Intentar buscar en src/datos_entrada primero, luego en el directorio raíz del proyecto
    datos_entrada_path = os.path.join(src_dir, 'datos_entrada')
    if not os.path.exists(datos_entrada_path):
        root_datos_path = os.path.join(project_root, 'datos_entrada')
        if os.path.exists(root_datos_path):
            datos_entrada_path = root_datos_path
            
    raw_matrices = load_midi_dataset(datos_entrada_path)
    
    # 2. Preparar secuencias y verificar si hay suficientes datos
    X, y = prepare_sequences(raw_matrices, SEQ_LENGTH)
    
    if len(X) == 0:
        print("\nNo se encontraron suficientes secuencias de longitud válida para entrenar en 'datos_entrada'.")
        print("Generando un archivo MIDI largo de respaldo...")
        create_fallback_midi(os.path.join(datos_entrada_path, 'ejemplo_escala_mayor.mid'))
        # Re-cargar
        raw_matrices = load_midi_dataset(datos_entrada_path)
        X, y = prepare_sequences(raw_matrices, SEQ_LENGTH)
    
    print(f"Dimensiones de entrenamiento: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("No hay suficientes datos para entrenar.")
        exit()
        
    # 3. Construir y Entrenar
    # Intentamos cargar la arquitectura desde el archivo de texto
    default_file = os.path.join(os.path.dirname(__file__), 'arquitectura_tensor.txt')
    alt_file = os.path.join(os.path.dirname(__file__), 'arquitactura_tensor.txt')
    target_file = default_file if os.path.exists(default_file) else (alt_file if os.path.exists(alt_file) else None)
    
    if target_file:
        print(f"Cargando configuración de red desde {target_file}...")
        estructura_red = network_builder.load_architecture_from_file(target_file)
    else:
        # Fallback en caso de que no existan los archivos de arquitectura
        estructura_red = [
            (network_builder.LAYER_CONV1D, 32, 2, 'relu'),
            (network_builder.LAYER_LSTM, 64, 'relu'),
            (network_builder.LAYER_DROPOUT, 0.2),
            (network_builder.LAYER_DENSE, 32, 'relu')
        ]
        
    model = build_model((SEQ_LENGTH, NUM_FEATURES), structure=estructura_red)
    model.summary()
    
    print(f"Entrenando por {EPOCHS} épocas...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 4. Generar
    # Usamos la primera secuencia de los datos como semilla
    seed = X[0]
    new_melody_matrix = generate_melody(model, seed, length=16)
    
    print("\nMatriz Generada Cruda (primeras 5 filas):")
    print(new_melody_matrix[:5])
    
    # 5. Guardar MIDI Crudo
    output_file_raw = "melodia_generada_ia_cruda.mid"
    midi_processor.matrix_to_midi(new_melody_matrix, output_file_raw, start_note=60)
    print(f"Melodía original (cruda) guardada en '{output_file_raw}'.")
    
    # 6. Aplicar capa de Árbol de Decisión
    # Parámetros: tonic=1 (C), greek_mode=2 (Dórico), max_skips=2, max_leaps=1, max_non_diatonic=1
    tonic_val = 1
    greek_mode_val = 2
    max_skips_val = 2
    max_leaps_val = 1
    max_non_diatonic_val = 1
    
    print(f"\nAplicando filtro de árbol de decisión con parámetros:")
    print(f"  - Tónica: {tonic_val} (C)")
    print(f"  - Modo Griego: {greek_mode_val} (Dórico)")
    print(f"  - Max Skips: {max_skips_val}")
    print(f"  - Max Leaps: {max_leaps_val}")
    print(f"  - Max No Diatónicas: {max_non_diatonic_val}")
    
    filtered_melody_matrix = decision_tree_filter.apply_decision_tree_filter(
        new_melody_matrix,
        tonic=tonic_val,
        greek_mode=greek_mode_val,
        max_skips=max_skips_val,
        max_leaps=max_leaps_val,
        max_non_diatonic=max_non_diatonic_val,
        start_note=60
    )
    
    # 7. Guardar MIDI Filtrado por Árbol de Decisión
    output_file_filtered = "melodia_generada_ia_filtrada.mid"
    midi_processor.matrix_to_midi(filtered_melody_matrix, output_file_filtered, start_note=60)
    print(f"\n¡Listo! Melodías guardadas:\n  - Cruda: '{output_file_raw}'\n  - Filtrada (Árbol de Decisión): '{output_file_filtered}'")
