# HPC-IA-Tarea

## Actividades:

### 1: Conexion al Patagon:
*(Saltar hasta iii si ya tiene cuenta en Patagon)*
- 1. Configurar software SSH a usar con par de llaves publica/privada.
- 2. Enviar llave publica a felipe.quezada01@outlook.com para agregarla al usuario que usaran durante la actividad
- 3. Descargar el contenedor a utilizar e instalar las librerias


# 2: 

- 1. Leer y Ejecutar el programa tarea.py en su contenedor poniendo especial atencion en el tiempo de ejecucion.
- 2.  Cambiar parametros num_workers, pin_memory, prefetch_factor, batch size y ver como afectan en el tiempo.
- 3.  Ahora entrenar con una GPU y repetir el proceso. Como se ve afectado el tiempo de entrenamiento? qué hay de los resultados? ¿Por qué?
- 4. Utilizar DataParallel para entrenar con 2 GPUs y repetir el proceso, Cuál es la diferencia con respecto a CPU, y una GPU?

# 3: 

- 1. Modificar el codigo para utilizar la libreria Datasets de Hugging Face y cargar el dataset tiny-imagenet https://huggingface.co/datasets/Maysee/tiny-imagenet
- 2. Agregar cacheo de datos y al menos una tecnica de aumentacion de datos.
- 3. Entrenar el modelo con parametros encontrados en la actividad anterior y utilizando DataParallel. Medir una aproximacion del tiempo de ejecucion.
- 4. Entrenar el modelo con distributed data parallel ahora, y comparando los resultados con iii). ¿Cuál es la diferencia? ¿Que hay sobre la complejidad en la implementacion?

