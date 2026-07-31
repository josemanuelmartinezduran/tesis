# Archivo de Contexto y Reglas de Antigravity: Tesis Doctoral (Capítulo 3)

## 1. Identificación del Proyecto y Contexto General

- **Título de la Tesis**: *Inteligencia artificial generativa paramétrica aplicada a la composición musical*
- **Autor**: José Manuel Martínez Durán, MVR / DSC
- **Programa**: Doctorado en Ingeniería en Sistemas Computacionales (División de Posgrado, Universidad Da Vinci)
- **Director de Tesis**: Dr. Agustín Santiago Alvarado
- **Comité de Disertación**: Dr. Pedro Damián Reyes, Dr. José Aníbal Arias Aguilar
- **Fecha de desarrollo**: 2026
- **Propósito de este Workspace**: Desarrollo, redacción, formalización matemática e implementación del **Capítulo 3: Metodología y Diseño de la Arquitectura Híbrida**.

---

## 2. Definición del Problema y Objetivos Técnicos

### Problema Central
Los sistemas actuales de Inteligencia Artificial para la composición musical operan mayoritariamente como "cajas negras" (v.gr., generadores de audio de extremo a extremo o modelos de lenguaje secuenciales absolutos), impidiendo el control determinista, la parametrización de variables teóricas (tonalidad, modo, métrica) y la edición profesional en formato simbólico. Además, tienden a generar violaciones contrapuntísticas y armónicas que vulneran la coherencia musical.

### Propuesta Metodológica (Arquitectura Híbrida)
Evaluar y desarrollar una arquitectura híbrida de IA que combina:
1. **Representación Matricial Relativa**: Transformación de eventos musicales en vectores interválicos relativos referenciados a la tonalidad y a la nota anterior, normalizados para aprendizaje profundo.
2. **Núcleo Generativo (Redes Neuronales / Aprendizaje Profundo)**: Modelo generativo de melodías y estructuras armónicas representadas en formato matricial.
3. **Filtros Deterministas (Árboles de Decisión / Reglas de Contrapunto)**: Verificación y poda mediante reglas duras de contrapunto estricto (1ra a 4ta especie), controlando tonalidad, modo y métrica.
4. **Optimización Evolutiva (Algoritmos Genéticos)**: Retroalimentación basada en preferencias del usuario/compositor para evolucionar los parámetros genéticos vinculados a las capas paramétricas de la red neuronal.
5. **Pipeline de Salida**: Conversión de vectores validados a archivos MIDI editables de alta fidelidad.

---

## 3. Estructura Detallada del Capítulo 3

Cualquier adición o modificación al Capítulo 3 debe alinearse rigurosamente con la siguiente estructura capitular ([indice.txt](file:///home/data/Documentos/Doctorado/Tesis%20III/indice.txt)):

- **3.1 Enfoque de la Investigación**
  - 3.1.1 Tipo de estudio: Investigación y Desarrollo (I+D)
  - 3.1.2 Fases del desarrollo del sistema
- **3.2 Formalización de la Representación Matricial Relativa**
  - 3.2.1 Abstracción de datos: Del evento absoluto al vector relativo
  - 3.2.2 Algoritmo de mapeo de funciones armónicas y distancias interválicas
  - 3.2.3 Normalización y preparación del dataset para el entrenamiento
- **3.3 Diseño del Núcleo Generativo**
  - 3.3.1 Arquitectura de la red neuronal
  - 3.3.2 Capas paramétricas enlazadas a genes
  - 3.3.3 Definición de la capa de salida
- **3.4 Implementación de Filtros Deterministas**
  - 3.4.1 Codificación de reglas de contrapunto estricto (1ra a 4ta especie)
  - 3.4.2 Lógica de poda y validación de secuencias candidatas
  - 3.4.3 Interfaz de parametrización: Tonalidad, Modo y Métrica
- **3.5 Optimización Evolutiva**
  - 3.5.1 Definición del cromosoma: Representación de la estructura musical y estilo
  - 3.5.2 Operadores genéticos: Cruce y mutación aplicados a la notación simbólica
- **3.6 Integración de la Arquitectura y Workflow**
  - 3.6.1 Comunicación entre módulos
  - 3.6.2 Pipeline de salida: Generación del archivo MIDI editable
- **3.7 Estrategia de Validación y Pruebas**
  - 3.7.1 Definición de métricas de error armónico
  - 3.7.2 Diseño de los experimentos comparativos (Sistema propuesto vs. Modelos tradicionales)

---

## 4. Estándares de Redacción y Formato LaTeX

- **Estilo y Tono**: Lenguaje académico formal, impersonal, característico de disertación doctoral en ingeniería y computación.
- **Archivo Principal**: `capitulos 1-3.tex` ([capitulos 1-3.tex](file:///home/data/Documentos/Doctorado/Tesis%20III/capitulos%201-3.tex))
- **Módulos TeX Asociados**:
  - `conversor.tex` (Representación Matricial Relativa)
  - `neuronal.tex` (Núcleo Generativo)
  - `arboles_decision.tex` (Filtros Deterministas)
  - `geneticos.tex` (Optimización Evolutiva)
  - `workflow.tex` / `explicacion.tex` (Workflow e Integración)
- **Motor LaTeX y Paquetes**: `\documentclass[12pt]{report}`, `babel` en español (`spanish,es-nodecimaldot,es-noshorthands`), `amsmath`, `amssymb`, `graphicx`, `hyperref`, `csquotes`, `float`.
- **Gestión Bibliográfica**: Uso estricto de BibTeX con la base de datos `capitulo1-3.bib` ([capitulo1-3.bib](file:///home/data/Documentos/Doctorado/Tesis%20III/capitulo1-3.bib)). Todas las afirmaciones teóricas y antecedentes deben incluir sus respectivas citas (`\cite{...}`).
- **Formalización Matemática**: Expresar ecuaciones, vectores, funciones de adecuación (fitness) y espacios de estados mediante sintaxis limpia de `amsmath`.

---

## 5. Directrices de Acción para el Agente Antigravity

1. **Rigor Académico y Científico**: Priorizar explicaciones formales, formulaciones matemáticas claras, pseudocódigo estructurado y diagramas explicativos.
2. **Modularidad**: Al editar contenido del Capítulo 3, mantener la división en archivos `.tex` modulares incluidos o compilados en el documento maestro `capitulos 1-3.tex`.
3. **Verificación de Compilación**: Tras realizar cambios significativos en los archivos `.tex`, ejecutar la compilación de prueba (v.gr., `pdflatex -interaction=nonstopmode "capitulos 1-3.tex"`) y revisar el archivo `.log` para asegurar cero errores críticos de compilación o referencias rotas.
4. **Cuidado de la Bibliografía**: Asegurarse de mantener las entradas BibTeX bien estructuradas en `capitulo1-3.bib` sin duplicados.
5. **No Suposiciones**: En caso de duda sobre especificaciones del algoritmo o requerimientos del comité de disertación, consultar formalmente al usuario antes de modificar postulados teóricos centrales.
