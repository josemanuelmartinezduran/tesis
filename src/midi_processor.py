import mido
import math
import numpy as np

def note_name_to_number(name):
    """Convierte nombre de nota (ej: 'C', 'G#') a número 0-11."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes.index(name.upper())

def get_scale_degree(pitch_class, root_note, scale_intervals):
    """
    Calcula la posición relativa en la escala.
    pitch_class: 0-11 (nota actual)
    root_note: 0-11 (tonica)
    scale_intervals: lista de intervalos acumulados desde la tónica (ej: Mayor = [0, 2, 4, 5, 7, 9, 11])
    
    Retorna:
    - 1.0, 2.0, 3.0... para notas diatónicas.
    - 1.5, 2.5... para notas cromáticas intermedias.
    """
    # Normalizar la nota actual relativa a la tónica
    relative_pitch = (pitch_class - root_note) % 12
    
    if relative_pitch in scale_intervals:
        return float(scale_intervals.index(relative_pitch) + 1)
    
    for i in range(len(scale_intervals) - 1):
        lower = scale_intervals[i]
        upper = scale_intervals[i+1]
        if lower < relative_pitch < upper:
            return float((i + 1) + 0.5)
            
    if relative_pitch > scale_intervals[-1]:
         return float(len(scale_intervals) + 0.5)
         
    return 0.0 # Fallback

def midi_to_matrix(midi_path, root_note_name='C', scale_type='major'):
    """
    Convierte un MIDI a la matriz numérica especificada.
    """
    mid = mido.MidiFile(midi_path)
    
    if scale_type == 'major':
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        raise ValueError(f"Tipo de escala '{scale_type}' no soportado.")
    
    root = note_name_to_number(root_note_name)
    
    matrix_list = []
    
    last_note_pitch = None
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_unit = ticks_per_beat / 4.0
    
    merged_events = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                merged_events.append((abs_time, msg))
    
    merged_events.sort(key=lambda x: x[0])
    
    final_notes = [] 
    active_notes = {} 
    
    for abs_tick, msg in merged_events:
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = abs_tick
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                start_tick = active_notes.pop(msg.note)
                duration = abs_tick - start_tick
                final_notes.append({
                    'pitch': msg.note,
                    'duration': duration,
                    'start': start_tick
                })
    
    final_notes.sort(key=lambda x: x['start'])
    
    for i, note in enumerate(final_notes):
        row = []
        midi_pitch = note['pitch']
        
        pitch_class = midi_pitch % 12
        row.append(float(pitch_class))
        
        duration_units = note['duration'] / ticks_per_unit
        row.append(round(duration_units, 2))
        
        degree = get_scale_degree(pitch_class, root, scale_intervals)
        row.append(degree)
        
        if last_note_pitch is None:
            interval = 0.0
        else:
            semitones = midi_pitch - last_note_pitch
            interval = semitones / 2.0
        
        row.append(interval)
        
        matrix_list.append(row)
        
        last_note_pitch = midi_pitch
        
    return np.array(matrix_list)

def matrix_to_midi(matrix, output_path='output.mid', start_note=60, bpm=120):
    """
    Reconstruye un archivo MIDI a partir de la matriz numérica.
    Usa la columna de 'Intervalo' para calcular la altura de las notas.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Configuración de tiempo
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    # Tempo: microsegundos por beat = 60,000,000 / bpm
    tempo = int(60000000 / bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # 1/16 de nota en ticks (Duration = 1.0 en la matriz)
    ticks_per_unit = int(ticks_per_beat / 4)
    
    current_pitch = start_note
    
    # La matriz tiene: [PitchClass, Duration, Degree, Interval]
    
    for row in matrix:
        # 1. Calcular Duración
        duration_val = max(0.1, float(row[1])) # Evitar duracion 0 o negativa
        duration_ticks = int(duration_val * ticks_per_unit)
        
        # 2. Calcular Pitch
        # Usamos la columna 3 (Intervalo) para movernos desde la nota anterior
        # Intervalo está en Tonos. 1 Tono = 2 Semitonos (MIDI pitch units)
        interval_tones = float(row[3])
        semitones_delta = int(round(interval_tones * 2))
        
        # Aplicar intervalo (La primera nota usa el intervalo tal cual, 
        # asumimos que la fila 0 tiene intervalo 0 o relativo a la 'nada')
        current_pitch += semitones_delta
        
        # Clampear pitch a rango MIDI válido (0-127)
        mid_pitch = max(0, min(127, int(current_pitch)))
        
        # 3. Crear eventos MIDI
        # Nota Encendida (Note On)
        track.append(mido.Message('note_on', note=mid_pitch, velocity=64, time=0))
        # Nota Apagada (Note Off - el tiempo delta va aquí para dar la duración)
        track.append(mido.Message('note_off', note=mid_pitch, velocity=64, time=duration_ticks))
        
    mid.save(output_path)
    print(f"Archivo guardado: {output_path}")

def create_test_midi(filename='test_output.mid'):
    """Crea un archivo MIDI simple: Escala de Do Mayor con una nota cromática."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Ticks por pulso por defecto es usualmente 480
    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    # Nota de un dieciseisavo (semicorchea) = 120 ticks
    sixteenth = int(ticks_per_beat / 4)
    
    # Notas: C, D, D#, E (Do, Re, Re# (no diatónico), Mi)
    # Duraciones: 1/16, 1/8 (2/16), 1/16, 1/4 (4/16)
    notes = [
        (60, sixteenth),      # C4, dura 1
        (62, sixteenth * 2),  # D4, dura 2
        (63, sixteenth),      # D#4, dura 1 (Cromático en Do Mayor)
        (64, sixteenth * 4)   # E4, dura 4
    ]
    
    current_time = 0
    for pitch, duration in notes:
        # Nota Encendida (Note On)
        track.append(mido.Message('note_on', note=pitch, velocity=64, time=0))
        # Nota Apagada (Note Off - tiempo delta = duración)
        track.append(mido.Message('note_off', note=pitch, velocity=64, time=duration))
        
    mid.save(filename)
    print(f"Archivo MIDI de prueba creado: {filename}")

if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description='Convertir MIDI a Matriz Numérica (4 columnas).')
    parser.add_argument('file', nargs='?', help='Ruta al archivo MIDI de entrada')
    parser.add_argument('--root', default='C', help='Nota tónica de la escala (ej: C, G, F#). Por defecto: C')
    parser.add_argument('--scale', default='major', help='Tipo de escala (major). Por defecto: major')
    parser.add_argument('--test', action='store_true', help='Generar y procesar archivo de prueba')
    parser.add_argument('--convert-back', help='Ruta para guardar el MIDI reconvertido (prueba inversa)')

    args = parser.parse_args()

    # Si se pide test o no hay archivo, generar test
    if args.test or not args.file:
        test_file = 'melodia_prueba.mid'
        create_test_midi(test_file)
        target_file = test_file
        print(f"--- Modo Prueba: Usando {test_file} ---")
    else:
        target_file = args.file

    if not os.path.exists(target_file):
        print(f"Error: El archivo '{target_file}' no existe.")
        sys.exit(1)

    print(f"Procesando: {target_file} | Tonalidad: {args.root} {args.scale}")
    
    try:
        result_matrix = midi_to_matrix(target_file, root_note_name=args.root, scale_type=args.scale)
        
        print("\nMatriz Resultante:")
        print(f"{'Nota':<6} {'Dur':<6} {'Grado':<6} {'Intervalo':<8}")
        print("-" * 30)
        for row in result_matrix:
            # Formato limpio: Nota(int), Dur(float), Grado(float), Int(float)
            print(f"{int(row[0]):<6} {row[1]:<6} {row[2]:<6} {row[3]:<8}")
            
        if args.convert_back:
            print(f"\nReconstruyendo MIDI en: {args.convert_back}")
            # Usamos la primera nota del MIDI original si es posible, sino 60 (Do central)
            # Como la matriz no guarda la octava absoluta, el resultado puede estar transpuesto.
            matrix_to_midi(result_matrix, output_path=args.convert_back, start_note=60)
            
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
        print("Asegúrate de ejecutar esto en un entorno con 'mido' instalado.")
