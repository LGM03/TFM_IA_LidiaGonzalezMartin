## Resumen del proyecto

Este repositorio contiene la implementación experimental del Trabajo Fin de Máster (TFM) orientado al análisis comparativo entre enfoques clásicos de aprendizaje automático y arquitecturas modernas de Deep Learning aplicadas a la clasificación de imágenes.

El estudio evalúa el rendimiento, la eficiencia computacional y la explicabilidad de los siguientes modelos:

* Baseline clásico: HOG + SVM
* ResNet50
* EfficientNet-B3
* Vision Transformer (ViT-B/16)

Para la reproductividad del trabajo se recomienda descargar el dataset RAF-DB de Kaggle (https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) y agregarlo al directorio principal.

Por restricciones de tamaño, los modelos entrenados se encuentran disponibles através de Drive (https://drive.google.com/drive/folders/19xTCW5RG9nLwS2fdLhvgyftNhzk9jqXt?usp=drive_link), para el correcto funcionamiento del codigo, se recomienda guardar cada modelo en su correspondiente carpeta. Por ejemplo, la carpeta vit_fase2_final, debe ir dentro de la carpeta vit.

## Objetivos

Los objetivos principales del trabajo son:

* Comparar métodos clásicos y modernos de clasificación de emociones tras un proceso de Transfer Learning.
* Analizar el impacto de la arquitectura en el rendimiento predictivo.
* Evaluar la eficiencia computacional en CPU y GPU.
* Estudiar la explicabilidad de los modelos mediante analisis de Grad-CAM, Attention Rollout, representaciones del espacio vectorial, prototipos y críticas
* Analizar la los resultados en concepto de complejidad del modelo, rendimiento y latencia.

Para contexto metodológico, análisis de resultados y conclusiones, consulta TFM_LidiaGonzalezMartin.pdf

## Modelos utilizados

### Modelo clásico

* HOG + SVM como baseline tradicional de visión por computador.

### Modelos de Deep Learning

* **ResNet50** : red neuronal convolucional con conexiones residuales.
* **EfficientNet-B3** : arquitectura optimizada mediante escalado compuesto.
* **ViT-B/16** : modelo basado en Transformers aplicado a visión por computador.

## Estructura del repositorio

```
TFM_IA_LidiaGonzalezMartin/
│
├── efficientnet/
├── resnet/
├── vit/
├── hog+svm/
│
├── candidatas_finales_comunes.csv
├── fallos_comunes.csv
├── seleccion_imagenes.py
│
├── TFM.docx
└── README.md
```

Cada carpeta contiene:

* Notebooks de entrenamiento y evaluación
* Resultados experimentales
* Análisis de explicabilidad
* Visualizaciones y métricas

## Reproducibilidad

Para garantizar la reproducibilidad de los experimentos:

* Uso de semillas fijas en los distintos frameworks.
* División consistente del conjunto de datos.
* Pipeline de preprocesamiento estandarizado.
* Los modelos entrenados y el dataset utilizado no se incluyen en el repositorio debido a su tamaño.

Para garantizar la reproducibilidad se proporcionan tres ficheros requirements.txt. Se recomienda la creación de tres entornos virtuales diferenciados para la ejecución de cada fichero.

* **requirements_vit.txt:** Contiene las dependencias necesarias para ejecutar los notebooks de entrenamiento de ViT-B/16 y su explicabilidad.
* **requirements_cnns.txt:** Contiene las dependencias necesarias para ejecutar los notebooks de entrenamiento de EfficientNetB3 y ResNet50 y su explicabilidad.
* **requirements_hog.txt:** Contiene las dependencias necesarias para ejecutar los notebooks de entrenamiento del baseline HOG+SVM y su explicabilidad.
