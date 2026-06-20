import os
import shutil
from datetime import datetime
import numpy as np
import midi_processor

# Definición de rutas
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SRC_DIR, 'matrices_entrada')
HISTORY_DIR = os.path.join(INPUT_DIR, 'historial')

# Intervalos de Modos Griegos
MODE_INTERVALS = {
    1: [0, 2, 4, 5, 7, 9, 11], # Jónico (Mayor)
    2: [0, 2, 3, 5, 7, 9, 10], # Dórico
    3: [0, 1, 3, 5, 7, 8, 10], # Frigio
    4: [0, 2, 4, 6, 7, 9, 11], # Lidio
    5: [0, 2, 4, 5, 7, 9, 10], # Mixolidio
    6: [0, 2, 3, 5, 7, 8, 10], # Eólico (Menor Natural)
    7: [0, 1, 3, 5, 6, 8, 10]  # Locrio
}

# Nombres de notas para mapeo de tónica
TONIC_NAMES = {
    1: 'C', 2: 'C#', 3: 'D', 4: 'D#', 5: 'E', 6: 'F',
    7: 'F#', 8: 'G', 9: 'G#', 10: 'A', 11: 'A#', 12: 'B'
}

def save_matrix_to_file(matrix, filepath):
    """Guarda una matriz en un archivo de texto separado por comas."""
    with open(filepath, 'w') as f:
        f.write("# PitchClass, Duration, Degree, Interval\n")
        for row in matrix:
            f.write(", ".join(f"{val:.2f}" for val in row) + "\n")

def load_matrix_from_file(filepath):
    """Carga una matriz desde un archivo de texto."""
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            row = [float(val.strip()) for val in line.split(',')]
            matrix.append(row)
    return np.array(matrix)

def generate_random_individual(length, root_note, scale_intervals, start_note=60):
    """
    Genera un individuo melódico inicial válido.
    Garantiza consistencia entre notas, grados y relaciones interválicas.
    """
    diatonic_notes = [(root_note + interval) % 12 for interval in scale_intervals]
    durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    matrix_list = []
    last_pitch = start_note
    
    for _ in range(length):
        row = []
        # Nota diatónica aleatoria
        pitch_class = int(np.random.choice(diatonic_notes))
        row.append(float(pitch_class))
        
        # Duración aleatoria
        dur = float(np.random.choice(durations))
        row.append(dur)
        
        # Grado diatónico
        deg = midi_processor.get_scale_degree(pitch_class, root_note, scale_intervals)
        row.append(deg)
        
        # Intervalo respecto al anterior (intentando mantener saltos suaves)
        test_pitch = pitch_class + 12 * (last_pitch // 12)
        # Clampear para evitar registros demasiado lejanos
        if abs(test_pitch - last_pitch) > 6:
            test_pitch = test_pitch - 12 if test_pitch > last_pitch else test_pitch + 12
        interval = (test_pitch - last_pitch) / 2.0
        row.append(interval)
        
        matrix_list.append(row)
        last_pitch = test_pitch
        
    return np.array(matrix_list)

def crossover(parent_a, parent_b):
    """Realiza un cruce en un punto aleatorio entre dos matrices."""
    len_a = len(parent_a)
    len_b = len(parent_b)
    min_len = min(len_a, len_b)
    if min_len <= 2:
        return parent_a.copy()
    point = np.random.randint(1, min_len - 1)
    child = np.vstack([parent_a[:point], parent_b[point:]])
    return child

def mutate_melody(matrix, root_note, scale_intervals, prob_mut=0.15):
    """
    Introduce mutaciones aleatorias con coherencia teórica:
    - Mutación de pitch class: re-calcula su grado y ajusta intervalos locales
      para que el resto de la melodía no sufra transposiciones no deseadas.
    - Mutación de duración: asigna valores de duración rítmica convencionales.
    """
    mutated = matrix.copy()
    n_notes = len(mutated)
    durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    for j in range(n_notes):
        if np.random.rand() < prob_mut:
            if np.random.rand() > 0.5:
                # Mutar Nota (Pitch Class)
                old_pc = int(round(mutated[j][0]))
                diatonic_notes = [(root_note + interval) % 12 for interval in scale_intervals]
                new_pc = int(np.random.choice(diatonic_notes))
                
                if new_pc != old_pc:
                    diff = new_pc - old_pc
                    # Mover por la distancia más corta en la octava
                    if diff > 6: diff -= 12
                    elif diff <= -6: diff += 12
                    
                    mutated[j][0] = float(new_pc)
                    mutated[j][2] = midi_processor.get_scale_degree(new_pc, root_note, scale_intervals)
                    
                    # Ajustar intervalos locales (mutación localizada)
                    mutated[j][3] += diff / 2.0
                    if j + 1 < n_notes:
                        mutated[j+1][3] -= diff / 2.0
            else:
                # Mutar Duración
                mutated[j][1] = float(np.random.choice(durations))
                
    return mutated

def get_melody_string(matrix, root_note, scale_intervals):
    """Devuelve una representación legible de las notas del individuo."""
    notes_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    melody = []
    for row in matrix:
        pc = int(round(row[0]))
        dur = row[1]
        melody.append(f"{notes_names[pc]}({dur})")
    return " -> ".join(melody[:6]) + ("..." if len(matrix) > 6 else "")

def main():
    print("=" * 60)
    print("      ALGORITMO GENÉTICO INTERACTIVO PARA MELODÍAS      ")
    print("=" * 60)
    
    # Crear carpetas si no existen
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Parámetros del entorno armónico
    try:
        print("\nSelecciona los parámetros de la escala:")
        tonic = int(input("Tónica (1=C, 2=C#, ..., 12=B) [Por defecto: 1 (C)]: ") or 1)
        greek_mode = int(input("Modo Griego (1=Jónico, 2=Dórico, 3=Frigio, 4=Lidio, 5=Mixolidio, 6=Eólico, 7=Locrio) [Por defecto: 2 (Dórico)]: ") or 2)
    except ValueError:
        print("Entrada inválida. Usando valores predeterminados (Do Dórico).")
        tonic = 1
        greek_mode = 2
        
    root_note = (tonic - 1) % 12
    scale_intervals = MODE_INTERVALS.get(greek_mode, [0, 2, 4, 5, 7, 9, 11])
    
    # Inicializar población si la carpeta está vacía
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith('individuo_') and f.endswith('.txt')]
    if len(files) == 0:
        print("\nLa carpeta 'matrices_entrada' está vacía. Generando población inicial (10 individuos)...")
        for idx in range(1, 11):
            ind = generate_random_individual(length=16, root_note=root_note, scale_intervals=scale_intervals)
            save_matrix_to_file(ind, os.path.join(INPUT_DIR, f'individuo_{idx}.txt'))
        files = [f for f in os.listdir(INPUT_DIR) if f.startswith('individuo_') and f.endswith('.txt')]
        
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Mostrar la población actual al usuario
    print("\nPoblación actual de melodías:")
    population = []
    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(INPUT_DIR, filename)
        matrix = load_matrix_from_file(filepath)
        population.append((filename, matrix))
        melody_str = get_melody_string(matrix, root_note, scale_intervals)
        print(f"  [{idx}] {filename:<18} | Notas: {melody_str}")
        
    # Selección interactiva
    user_input = input("\nIntroduce los números de las melodías que te gustaron (ej: 1,3,5): ")
    try:
        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip()]
        selected_parents = [population[i][1] for i in selected_indices if 0 <= i < len(population)]
    except Exception:
        selected_parents = []
        
    if len(selected_parents) == 0:
        print("No seleccionaste ningún individuo o la entrada fue inválida. Usando todos los individuos como padres.")
        selected_parents = [p[1] for p in population]
        
    # Guardar historial (hacer backup de la generación anterior)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_history_dir = os.path.join(HISTORY_DIR, f"generacion_{timestamp}")
    os.makedirs(gen_history_dir, exist_ok=True)
    for filename, _ in population:
        shutil.copy(os.path.join(INPUT_DIR, filename), os.path.join(gen_history_dir, filename))
    print(f"\nHistorial guardado en: matrices_entrada/historial/generacion_{timestamp}")
    
    # Generar 10 nuevos individuos (Cruce y Mutación)
    print("\nGenerando nueva generación (10 nuevos individuos)...")
    new_population = []
    for idx in range(1, 11):
        # Selección de padres aleatorios del grupo seleccionado
        parent_a = selected_parents[np.random.randint(0, len(selected_parents))]
        parent_b = selected_parents[np.random.randint(0, len(selected_parents))]
        
        # Cruce
        child = crossover(parent_a, parent_b)
        
        # Mutación
        child_mutated = mutate_melody(child, root_note, scale_intervals, prob_mut=0.15)
        
        new_population.append(child_mutated)
        
        # Guardar en archivo reemplazando los individuos actuales
        save_matrix_to_file(child_mutated, os.path.join(INPUT_DIR, f'individuo_{idx}.txt'))
        
    print("\n¡Listo! Se ha creado la nueva generación de 10 individuos en la carpeta 'matrices_entrada'.")
    print("Puedes volver a ejecutar este script para seguir evolucionando tus melodías.")

if __name__ == "__main__":
    main()
