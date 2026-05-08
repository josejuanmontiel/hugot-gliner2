# Tareas Pendientes (TODO) & Refactorización

Ahora que la matemática y la inferencia a nivel core (`math.go`, ONNX) funcionan y replican al 100% el comportamiento de PyTorch, el siguiente paso es empaquetar el código adecuadamente para que sea digerible, mantenible, y fácilmente integrable en `hugot` o cualquier otro sistema.

## 1. Ordenación y Empaquetado del Directorio

De cara a prepararlo como una librería nativa y limpiar la actual mezcla de scripts de desarrollo, se recomienda mover el código a la siguiente estructura idiomática en Go:

```text
hugot-gliner2/
├── cmd/
│   └── demo/                 # Binario final de ejemplo/CLI que inicialice el pipeline E2E
│       └── main.go
├── pkg/
│   └── gliner/               # Paquete principal para la librería
│       ├── pipeline.go       # Lógica central del Pipeline (inicialización y destrucción de recursos)
│       ├── math.go           # Álgebra lineal y construcción de spans (Gonum)
│       ├── safetensors.go    # Parser de tensores .safetensors
│       └── types.go          # Structs de configuración y respuestas (SpanMatch, RelationResult, etc.)
├── scripts/
│   └── export/               # Scripts de desarrollo para puenteo y exportación
│       ├── export_gliner.py  # Script de generación de ONNX
│       ├── bridge_processor.py # Generación de JSONs de simulación
│       └── test_*.py         # Tests de equivalencia matemática en python
├── tests/
│   ├── e2e_onnx_test.go      # El test de integración final usando los modelos
│   └── validate_math_test.go # Validación de paridad matemática
├── models/                   # (En gitignore)
│   ├── encoder.onnx
│   ├── count_embed.onnx
│   └── gliner_classifiers.safetensors
├── README.md
├── TODO.md
├── go.mod
└── go.sum
```

## 2. Refactorizaciones en Código Go

- **Encapsulación (`gliner.go`)**: 
  - Mover la inicialización de los bindings de ONNX (actualmente sucios y distribuidos en el test) a constructores de pipeline limpios (`NewPipeline(encoderPath, countEmbedPath, tensorsPath)`).
  - Gestionar correctamente los punteros a memoria `C` y el `defer Session.Close()`.
  
- **Abstracción del Pipeline**:
  - Limpiar el `e2e_onnx_test.go` para que consuma la API pública del paquete `gliner` en vez de implementar él mismo las llamadas base a ONNX.
  - Diseñar métodos claros de Inferencia pública: `ExtractEntities(hiddenStates)` y `ExtractRelations(hiddenStates, schemaEmbs)`.

## 3. Integración Final
- Adaptar las entradas de este motor a la respuesta que devuelve internamente la librería `hugot` tras usar su Tokenizer embebido en Rust.
- Optimizar memoria eliminando slices intermedios si es posible (reutilizando tensores estáticos donde Gonum y ONNX lo permitan).
