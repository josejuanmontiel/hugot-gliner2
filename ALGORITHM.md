# GLiNER2 Algoritmo de Extracción E2E (Go-Native)

Este documento detalla el algoritmo interno implementado en `pkg/gliner` para realizar la extracción conjunta de entidades y relaciones usando ONNX y Gonum en Go.

## 1. Tokenización y Word Pooling (`processor.go`)

A diferencia de los modelos Transformer estándar que asocian predicciones a subpalabras (tokens BPE), GLiNER requiere que la extracción se realice a nivel de **palabras completas**.

1. **División de Palabras**: El texto original se divide en palabras utilizando la expresión regular `[\p{L}\p{N}]+|[^\p{L}\p{N}\s]+`. Esto separa letras/números de signos de puntuación, manteniendo la integridad de las palabras.
2. **Tokenización BPE**: Todo el texto es tokenizado utilizando el `tokenizer.json` de HuggingFace, generando una lista de IDs de subpalabras y sus strings correspondientes (por ej. `▁Tech` y `Nova`).
3. **Alineación**: Iteramos simultáneamente sobre la lista de palabras limpias y la lista de subpalabras BPE. Acumulamos subpalabras hasta reconstruir la palabra original.
4. **Extracción de Índices**: Guardamos el índice de la *primera* subpalabra de cada palabra agrupada. Este vector (`text_word_indices`) es crucial para el paso de pooling.

## 2. Cross-Encoder Prompting (`pipeline.go`)

GLiNER es un modelo *Cross-Encoder* para la extracción de relaciones. Esto significa que la representación interna de las clases a predecir (el esquema) cambia dependiendo del contexto del texto.

1. **Prompt Estático**: Los IDs de los tokens correspondientes al esquema de relaciones están pre-calculados (por ej. `[P] trabaja en ( [R] head [R] tail )`).
2. **Concatenación**: Construimos una secuencia de entrada combinada: `[Prompt IDs] + [SEP_TEXT] + [Text IDs]`.
3. **Inferencia ONNX**: Pasamos esta secuencia larga por el modelo Transformer (Encoder). Dado el mecanismo de auto-atención, las palabras del texto prestan atención a las relaciones objetivo, y los tokens de relación (`[R]`) acumulan información semántica sobre el texto.

## 3. Extracción de Representaciones (`ExtractFromText`)

El modelo devuelve un tensor gigante (`last_hidden_state`). Lo dividimos en dos partes:

- **Embeddings de Relaciones (`pc_embs`)**: Buscamos las posiciones de los tokens `[R]` dentro del prompt y extraemos sus vectores (dimensión 768). Estos vectores ahora representan matemáticamente los conceptos de *origen* (head) y *destino* (tail) para cada etiqueta de relación.
- **Embeddings de Palabras**: Usando nuestro vector `text_word_indices`, saltamos a la posición de la primera subpalabra de cada palabra en el tensor y extraemos su representación (Word Pooling).

## 4. Matemáticas de Extracción (`math.go`)

A partir de los embeddings de palabras, utilizamos **Gonum** para reproducir las matemáticas exactas de PyTorch sin dependencias pesadas de Machine Learning:

### Extracción de Entidades
1. **Construcción de Spans**: Tomamos ventanas deslizantes de hasta `max_span_len=12` palabras, concatenando el embedding de la palabra inicial y final de cada ventana.
2. **Proyección (Feed Forward)**: Pasamos cada span por la red lineal de clasificación (cuyos pesos cargamos desde `gliner_classifiers.safetensors`).
3. **NMS (Non-Maximum Suppression)**: Filtramos los spans superpuestos reteniendo sólo los de mayor puntuación (logit > 0.5).

### Extracción de Relaciones
1. Obtenemos las representaciones empíricas de cada entidad candidata.
2. Proyectamos estas representaciones a un espacio semántico compartido mediante operaciones matriciales (`Bilinear` u operaciones de atención latentes).
3. Hacemos el **Producto Punto** entre la representación de la entidad proyectada y los `pc_embs` extraídos del paso 3.
4. Si la puntuación supera el umbral, confirmamos el par `(Head, Tail)` como una relación válida para esa clase.
