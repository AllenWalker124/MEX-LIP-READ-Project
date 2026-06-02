Este proyecto presenta un análisis del desempeño de varios modelos de aprendizaje profundo aplicados a la lectura de labios en el español de México. 

Ante la falta de conjuntos de datos específicos, se creó un dataset propio con hablantes nativos. Este dataset está clasificado en **vocales**, **consonantes** (por *punto* y *modo* de articulación) y **sinfones**.

Metodología:
- Dataset: Incluye grabaciones de letras y sílabas, procesadas mediante redimensión (1:1), recorte de la zona labial y extracción de fotogramas.
- Modelos: Se evaluaron arquitecturas de Redes Neuronales Convolucionales Tridimensionales (CNN 3D) y Redes Convolucionales Recurrentes de Largo Plazo (LRCN).
- Evaluación: Se utilizaron métricas de accuracy, precision, recall y F1 score, apoyadas por matrices de confusión. Como apoyo visual también se ocupó la técnica de gradCAM para poder visualizar qué partes o píxeles de una imagen fueron determinantes para que el modelo tomara una decisión.

Los resultados y conclusiones se encuentran detallados en el documento ***Reporte Técnico Final*** dentro de la carpeta con el mismo nombre. También se puede leer una versión resumida de los resultados en el ***Artículo Técnico***.

El conjunto de datos (dataset) **MEX-LIP-READ** aún no está terminado, pero la meta es publicar un artículo sobre este y liberarlo en cuanto se complete. Sabemos de antemano lo difícil que es conseguir un dataset de este estilo para trabajar la lectura de labios, y más aún para el Español mexicano. En caso de requerirlo, podemos enviarte una versión no terminada del dataset. Para solicitarlo, envía un correo a cualquiera de los que aparecen en el ***Artículo Técnico*** y con gusto te lo compartiremos.
