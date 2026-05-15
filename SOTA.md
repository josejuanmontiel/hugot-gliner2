# State of the Art (SOTA) en hugot-gliner2

Este documento describe las capacidades avanzadas de arquitectura y modularidad implementadas en `hugot-gliner2`, diseñadas para soportar investigación SOTA (State of the Art) y adaptabilidad a futuros modelos NLP, todo nativo en Go.

## 1. Arquitectura de Cabeceras Modulares (Pluggable Architecture)

A diferencia de los pipelines tradicionales fuertemente acoplados, `hugot-gliner2` abstrae la lógica de clasificación final mediante el patrón Strategy (`HeadArchitecture`). 

Esto permite inyectar diferentes configuraciones de red neuronal sobre el mismo *backbone* de extracción (ONNX) sin modificar el código base de inferencia.

### Beneficios:
* **Intercambio en Caliente (Hot-Swapping)**: Puedes instanciar un pipeline con arquitectura estándar (ReLU) y otro con SOTA (GELU) operando simultáneamente.
* **Aislamiento**: Nuevos experimentos matemáticos no rompen la estabilidad del modelo de producción.
* **Extensibilidad**: Preparado para futuras cabeceras complejas (ej. *Biaffine Attention* completa para extracción avanzada de relaciones).

## 2. Primitivas Matemáticas de Vanguardia (SOTA Math Engine)

El paquete `math` incluye implementaciones nativas (tanto en puro Go como optimizadas con `Gonum` vía build tags) de las funciones de activación más recientes utilizadas en arquitecturas como LLaMA 3 o Mistral:

* **GELU (Gaussian Error Linear Unit)**: Proporciona una no-linealidad más suave que ReLU, crucial para modelos transformer modernos.
* **SiLU / Swish**: Usado en arquitecturas avanzadas por sus propiedades de gradiente.
* **Biaffine Transformations (Fundamentos)**: Permite puntuar pares de tokens interactuando de forma bilineal, esencial para la extracción de relaciones complejas en grafos de conocimiento.

## 3. Fine-Tuning y Entrenamiento Nativo en Go (Gorgonia Integration)

Una de las características más potentes de este framework es su capacidad para **entrenar cabeceras especializadas (Domain Experts)** sin salir del ecosistema Go, eliminando la dependencia tradicional de Python (PyTorch/TensorFlow).

### El Bucle de Vida Completo de la IA:
1. **Extracción de Características (ONNX)**: Se utiliza el Transformer base (`encoder.onnx`) para generar embeddings de texto de alta fidelidad (768 dimensiones).
2. **Entrenamiento (Gorgonia)**: Mediante `gorgonia.org/gorgonia`, se construyen grafos computacionales para entrenar capas de clasificación (ej. Detección de Síntomas Médicos) usando optimizadores como Adam y calculando gradientes (Backpropagation) sobre los embeddings reales de ONNX.
3. **Exportación Estándar**: Los pesos entrenados en Go se exportan dinámicamente al formato estándar de la industria `.safetensors`.
4. **Carga Dinámica**: A través de arquitecturas personalizadas (ej. `MedicalArchitecture`), el pipeline principal (`NewPipelineWithArch`) puede cargar estos pesos sobre la marcha y aplicarlos inmediatamente a la inferencia.

### Ejemplo de Flujo de Expertos de Dominio:
Puedes revisar `cmd/train/main.go` para ver un ejemplo funcional de entrenamiento de un "Medical Expert". Aunque un modelo preciso requiere un dataset masivo, la infraestructura soporta el entrenamiento de clasificadores binarios especializados con total interoperabilidad dimensional.

---
*Construido para el futuro del procesamiento de lenguaje natural en Go.*
