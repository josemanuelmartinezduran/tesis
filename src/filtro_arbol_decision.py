import numpy as np
import procesador_midi as midi_processor

# Modos Griegos y sus intervalos acumulados desde la tónica (en semitonos)
MODE_INTERVALS = {
    1: [0, 2, 4, 5, 7, 9, 11], # Jónico (Escala Mayor)
    2: [0, 2, 3, 5, 7, 9, 10], # Dórico
    3: [0, 1, 3, 5, 7, 8, 10], # Frigio
    4: [0, 2, 4, 6, 7, 9, 11], # Lidio
    5: [0, 2, 4, 5, 7, 9, 10], # Mixolidio
    6: [0, 2, 3, 5, 7, 8, 10], # Eólico (Escala Menor Natural)
    7: [0, 1, 3, 5, 6, 8, 10]  # Locrio
}

def apply_decision_tree_filter(melody_matrix, tonic, greek_mode, max_skips, max_leaps, max_non_diatonic, start_note=60):
    """
    Aplica una capa de filtrado basada en un Árbol de Decisión para restringir y alterar 
    la melodía generada de acuerdo con las reglas y parámetros de la teoría musical:
    
    Parámetros:
      melody_matrix: Matriz generada por la red neuronal de forma (N, 4).
      tonic: Nota tónica de 1 a 12 (1=C, 2=C#, ..., 12=B).
      greek_mode: Código de 1 a 7 correspondiente al modo griego.
      max_skips: Número máximo permitido de intervalos de 'skips' (3-4 semitonos).
      max_leaps: Número máximo permitido de intervalos de 'leaps' (>= 5 semitonos).
      max_non_diatonic: Número máximo de notas fuera de la escala permitidas.
      start_note: Altura absoluta base de la primera nota (MIDI).
      
    Retorna:
      Una matriz NumPy de la misma forma que la entrada con los parámetros y notas corregidas.
    """
    # Mapear tónica a nota 0-11
    root_note = (tonic - 1) % 12
    scale_intervals = MODE_INTERVALS.get(greek_mode, [0, 2, 4, 5, 7, 9, 11])
    diatonic_notes = [(root_note + interval) % 12 for interval in scale_intervals]
    
    # 1. Reconstruir alturas absolutas MIDI del generador
    current_pitch = start_note
    absolute_pitches = []
    for row in melody_matrix:
        interval_tones = float(row[3])
        semitones_delta = int(round(interval_tones * 2))
        current_pitch += semitones_delta
        absolute_pitches.append(current_pitch)
        
    filtered_pitches = []
    non_diatonic_count = 0
    skip_count = 0
    leap_count = 0
    
    def get_nearest_diatonic(pitch):
        offset = 0
        while True:
            for sign in [1, -1]:
                test_pitch = pitch + sign * offset
                if (test_pitch % 12) in diatonic_notes:
                    return test_pitch
            offset += 1

    # 2. Procesamiento secuencial mediante Árbol de Decisión por Nota
    for i, pitch in enumerate(absolute_pitches):
        # Nodo 1: ¿Es una nota diatónica?
        is_diatonic = (pitch % 12) in diatonic_notes
        
        if not is_diatonic:
            # Nodo 2: ¿Hemos excedido el límite de notas no diatónicas?
            if non_diatonic_count >= max_non_diatonic:
                pitch = get_nearest_diatonic(pitch)
            else:
                non_diatonic_count += 1
                
        # La primera nota no tiene intervalo de transición
        if i == 0:
            filtered_pitches.append(pitch)
            continue
            
        prev_pitch = filtered_pitches[-1]
        interval = pitch - prev_pitch
        abs_interval = abs(interval)
        direction = 1 if interval >= 0 else -1
        
        # Nodo 3: ¿Es un salto corto / skip (3 o 4 semitonos)?
        if abs_interval in [3, 4]:
            # Nodo 4: ¿Hemos excedido el límite de skips?
            if skip_count >= max_skips:
                # Alterar nota para reducir el intervalo a un paso/step (1 o 2 semitonos)
                step_size = 2 if ((prev_pitch + direction * 2) % 12) in diatonic_notes else 1
                pitch = prev_pitch + direction * step_size
                if (pitch % 12) not in diatonic_notes:
                    pitch = get_nearest_diatonic(pitch)
            else:
                skip_count += 1
                
        # Nodo 5: ¿Es un salto largo / leap (>= 5 semitonos)?
        elif abs_interval >= 5:
            # Nodo 6: ¿Hemos excedido el límite de leaps?
            if leap_count >= max_leaps:
                # Intentar reducir a un skip (si aún quedan disponibles) o a un step
                if skip_count < max_skips:
                    test_pitch = prev_pitch + direction * 3
                    if (test_pitch % 12) in diatonic_notes:
                        pitch = test_pitch
                    else:
                        pitch = prev_pitch + direction * 4
                        if (pitch % 12) not in diatonic_notes:
                            pitch = get_nearest_diatonic(pitch)
                    skip_count += 1
                else:
                    # Forzar reducción a un step (1 o 2 semitonos)
                    step_size = 2 if ((prev_pitch + direction * 2) % 12) in diatonic_notes else 1
                    pitch = prev_pitch + direction * step_size
                    if (pitch % 12) not in diatonic_notes:
                        pitch = get_nearest_diatonic(pitch)
            else:
                leap_count += 1
                
        filtered_pitches.append(pitch)
        
    # 3. Reconstruir la matriz final a partir de las notas filtradas
    filtered_matrix = []
    last_pitch = start_note
    for i, pitch in enumerate(filtered_pitches):
        row = melody_matrix[i].copy()
        
        # Pitch Class
        row[0] = float(pitch % 12)
        
        # Grado de la escala (Degree)
        row[2] = midi_processor.get_scale_degree(pitch % 12, root_note, scale_intervals)
        
        # Intervalo (Interval)
        interval_semitones = pitch - last_pitch
        row[3] = interval_semitones / 2.0
        
        filtered_matrix.append(row)
        last_pitch = pitch
        
    print(f"\n[Filtro Árbol de Decisión] Procesado completado:")
    print(f"  - Notas no diatónicas permitidas: {non_diatonic_count}/{max_non_diatonic}")
    print(f"  - Skips (saltos de 3-4 semitonos) permitidos: {skip_count}/{max_skips}")
    print(f"  - Leaps (saltos >= 5 semitonos) permitidos: {leap_count}/{max_leaps}")
    
    return np.array(filtered_matrix)
