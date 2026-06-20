import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Conv2D, Conv1D, Flatten, Dropout, Reshape

# Definición de códigos de capas
LAYER_NONE = 0
LAYER_DENSE = 1
LAYER_CONV2D = 2
LAYER_LSTM = 3
LAYER_DROPOUT = 4
LAYER_FLATTEN = 5
LAYER_CONV1D = 6

# Mapeos numéricos para soportar matrices puras (ej. en algoritmos genéticos)
ACTIVATION_MAPPING = {
    0: 'linear',
    1: 'relu',
    2: 'tanh',
    3: 'sigmoid',
    4: 'softmax',
    5: 'selu',
    6: 'elu'
}

PADDING_MAPPING = {
    0: 'valid',
    1: 'same'
}

def build_dynamic_model(input_shape, layer_configs, output_dim=4):
    """
    Construye un modelo Sequential de Keras de manera dinámica basado en una lista,
    matriz (NumPy array) o tuplas de configuraciones de capas.
    
    Cada elemento/fila en 'layer_configs' puede ser:
      - Un entero (ej: 1, 3): Usa valores por defecto de la capa.
      - Una tupla/lista/numpy-row (ej: [1, 64, 1]): [código_capa, param1, param2, ...].
      - Un diccionario (ej: {'type': 3, 'units': 64, 'activation': 'tanh'}).
      
    Parámetros:
      input_shape: Tupla que indica la forma de entrada (ej: (SEQ_LENGTH, NUM_FEATURES)).
      layer_configs: Estructura matricial o lista que define las capas de la red.
      output_dim: Dimensión de la capa de salida final (regresión).
      
    Retorna:
      Un modelo Keras compilado con optimizador 'adam' y pérdida 'mse'.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for idx, config in enumerate(layer_configs):
        layer_type = None
        params = {}
        
        # 1. Parsear el tipo de configuración
        if isinstance(config, (int, float)):
            layer_type = int(config)
        elif isinstance(config, dict):
            if 'type' not in config:
                raise ValueError(f"El diccionario de configuración de la capa {idx} debe tener la clave 'type'.")
            layer_type = int(config['type'])
            params = {k: v for k, v in config.items() if k != 'type'}
        else:
            # Manejar numpy arrays, listas, tuplas u otros iterables de números
            config_list = []
            if hasattr(config, '__iter__') and not isinstance(config, str):
                config_list = [val.item() if hasattr(val, 'item') else val for val in config]
            else:
                config_list = [config]
                
            if len(config_list) == 0:
                raise ValueError(f"La configuración de la capa {idx} no puede estar vacía.")
                
            # Extraer el tipo de capa asegurando que sea entero
            raw_type = config_list[0]
            layer_type = int(raw_type.item() if hasattr(raw_type, 'item') else raw_type)
            
            # Helper para obtener parámetros de manera segura de la lista
            def get_param(lst, p_idx, default):
                if p_idx < len(lst):
                    val = lst[p_idx]
                    return val.item() if hasattr(val, 'item') else val
                return default
                
            # Decodificar activaciones y paddings si vienen como números
            def decode_activation(val, default='relu'):
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return ACTIVATION_MAPPING.get(int(val), default)
                return val

            def decode_padding(val, default='same'):
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return PADDING_MAPPING.get(int(val), default)
                return val

            if layer_type == LAYER_NONE:
                pass
            elif layer_type == LAYER_DENSE:
                params['units'] = int(get_param(config_list, 1, 32))
                raw_act = get_param(config_list, 2, 'relu')
                params['activation'] = decode_activation(raw_act, 'relu')
            elif layer_type == LAYER_LSTM:
                params['units'] = int(get_param(config_list, 1, 64))
                raw_act = get_param(config_list, 2, 'relu')
                params['activation'] = decode_activation(raw_act, 'relu')
                raw_ret_seq = get_param(config_list, 3, None)
                if raw_ret_seq is not None:
                    params['return_sequences'] = bool(raw_ret_seq)
            elif layer_type == LAYER_CONV2D:
                params['filters'] = int(get_param(config_list, 1, 16))
                k = get_param(config_list, 2, 2)
                if isinstance(k, (int, float)):
                    params['kernel_size'] = (int(k), int(k))
                else:
                    params['kernel_size'] = k
                raw_act = get_param(config_list, 3, 'relu')
                params['activation'] = decode_activation(raw_act, 'relu')
                raw_pad = get_param(config_list, 4, 'same')
                params['padding'] = decode_padding(raw_pad, 'same')
            elif layer_type == LAYER_CONV1D:
                params['filters'] = int(get_param(config_list, 1, 16))
                params['kernel_size'] = int(get_param(config_list, 2, 2))
                raw_act = get_param(config_list, 3, 'relu')
                params['activation'] = decode_activation(raw_act, 'relu')
                raw_pad = get_param(config_list, 4, 'same')
                params['padding'] = decode_padding(raw_pad, 'same')
            elif layer_type == LAYER_DROPOUT:
                params['rate'] = float(get_param(config_list, 1, 0.2))
            elif layer_type == LAYER_FLATTEN:
                pass
            
        # 2. Agregar la capa correspondiente al modelo Sequential
        if layer_type == LAYER_NONE:
            continue
            
        elif layer_type == LAYER_DENSE:
            units = params.get('units', 32)
            activation = params.get('activation', 'relu')
            model.add(Dense(units, activation=activation))
            
        elif layer_type == LAYER_CONV2D:
            # Conv2D espera entrada 3D (alto, ancho, canales).
            # Si venimos de una entrada 2D (seq_len, features), añadimos una dimensión de canal.
            current_shape = model.output_shape[1:]
            if len(current_shape) == 2:
                model.add(Reshape((current_shape[0], current_shape[1], 1)))
                
            filters = params.get('filters', 16)
            kernel_size = params.get('kernel_size', (2, 2))
            activation = params.get('activation', 'relu')
            padding = params.get('padding', 'same')
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding))
            
        elif layer_type == LAYER_LSTM:
            units = params.get('units', 64)
            activation = params.get('activation', 'relu')
            
            # Detectar automáticamente si necesitamos return_sequences=True
            # Si no está especificado por el usuario, buscamos si hay capas LSTM o recurrentes más adelante.
            return_seq = params.get('return_sequences')
            if return_seq is None:
                has_next_lstm = False
                for next_cfg in layer_configs[idx+1:]:
                    next_type = None
                    if isinstance(next_cfg, (int, float)):
                        next_type = int(next_cfg)
                    elif isinstance(next_cfg, dict):
                        next_type = next_cfg.get('type')
                    elif hasattr(next_cfg, '__iter__') and not isinstance(next_cfg, str):
                        next_list = list(next_cfg)
                        if len(next_list) > 0:
                            raw_next_type = next_list[0]
                            next_type = int(raw_next_type.item() if hasattr(raw_next_type, 'item') else raw_next_type)
                    if next_type == LAYER_LSTM:
                        has_next_lstm = True
                        break
                return_seq = has_next_lstm
                
            model.add(LSTM(units, activation=activation, return_sequences=return_seq))
            
        elif layer_type == LAYER_DROPOUT:
            rate = params.get('rate', 0.2)
            model.add(Dropout(rate))
            
        elif layer_type == LAYER_FLATTEN:
            model.add(Flatten())
            
        elif layer_type == LAYER_CONV1D:
            filters = params.get('filters', 16)
            kernel_size = params.get('kernel_size', 2)
            activation = params.get('activation', 'relu')
            padding = params.get('padding', 'same')
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding))
            
        else:
            raise ValueError(f"Código de capa desconocido: {layer_type}")
            
    # 3. Capa final de salida
    # Si la salida actual tiene más de una dimensión (excluyendo batch), aplanamos antes de Dense(output_dim).
    last_shape = model.output_shape[1:]
    if len(last_shape) > 1:
        model.add(Flatten())
        
    model.add(Dense(output_dim))
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='mse')
    return model

def load_architecture_from_file(filepath):
    """
    Carga la matriz de arquitectura de la red neuronal desde un archivo de texto.
    Cada línea del archivo representa una capa con sus parámetros numéricos separados por comas.
    Las líneas vacías o que comiencen con '#' son omitidas.
    """
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Separar por comas e intentar convertir a float
            row = [float(val.strip()) for val in line.split(',')]
            matrix.append(row)
    return matrix
