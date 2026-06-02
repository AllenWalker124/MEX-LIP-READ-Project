import os                   # Para manejar operaciones del sistema de archivos
# import cv2                  # OpenCV para procesamiento de imágenes
import numpy as np           # Numpy para manejo de arrays y operaciones numéricas
import random                # Para la generación de números aleatorios
import tensorflow as tf      # TensorFlow para construir y entrenar modelos de Deep Learning
import matplotlib.pyplot as plt  # Para visualización de datos y gráficos
from tensorflow.keras.models import Sequential, save_model  # Para la construcción y guardado de modelos
from tensorflow.keras.layers import (Conv3D, Conv2D, Flatten, Dense, MaxPooling3D, MaxPooling2D, TimeDistributed, LSTM, Dropout, BatchNormalization, Input, TimeDistributed)  # Capas utilizadas en el modelo
from tensorflow.keras.utils import Sequence, plot_model  # Utilidades para modelos y generación de secuencias de datos
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Callbacks para mejorar el entrenamiento
from tensorflow.keras.preprocessing.image import (img_to_array, load_img, smart_resize, ImageDataGenerator)  # Preprocesamiento de imágenes
from tensorflow.keras.optimizers import Adam     # Optimizador Adam para ajustar los pesos del modelo
from sklearn.model_selection import train_test_split  # Para dividir el dataset en entrenamiento y prueba
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score)  # Métricas para evaluar el modelo
from sklearn.utils import shuffle as sk_shuffle
from collections import Counter
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import tensorflow as tf

from tensorflow.keras.callbacks import Callback

class CustomDataGenerator(Sequence):
    def __init__(self, root_dir, mode='train', batch_size=32, frames_per_sample=18, target_size=(128, 128), n_channels=3, shuffle=False):
        """
        Inicializa el generador para una modalidad específica (train, validation o test).

        Parámetros:
        - root_dir: Ruta de la carpeta raíz que contiene los datos organizados por clasificación.
        - mode: Modalidad del generador ('train', 'validation', 'test').
        - batch_size: Tamaño del lote.
        - frames_per_sample: Número de frames por muestra.
        - target_size: Dimensiones de las imágenes (ancho y alto).
        - n_channels: Número de canales de color (3 para RGB).
        - shuffle: Indica si los datos deben mezclarse o no.
        """
        self.root_dir = root_dir  # Se pasa directamente la ruta de la clasificación
        self.mode = mode
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.target_size = target_size
        self.n_channels = n_channels
        self.shuffle = shuffle  # Establece si los datos deben mezclarse o no

        # Las clases ahora son las subcarpetas dentro del directorio raíz que representa la clasificación
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}

        self.dataset_info = self.prepare_dataset()
        self.samples = self.dataset_info[mode]

        # Si shuffle es True, mezcla las muestras al inicializar el generador
        if self.shuffle:
            self.samples = sk_shuffle(self.samples)

    def __len__(self):
        """
        Define la cantidad de lotes por época.
        """
        return len(self.samples) // self.batch_size

    def prepare_dataset(self):
        """
        Prepara el dataset dividiendo las muestras en train, validation y test.
        Asegura que cada subcarpeta tiene exactamente 68 muestras (o más, ajustable según el dataset).

        Retorna:
        - Diccionario con las muestras divididas en 'train', 'validation' y 'test'.
        """
        random.seed(42)  # Fija la semilla para asegurar consistencia en la división de datos
        all_files = {letter: [] for letter in self.classes}

        # Recorre cada subcarpeta (clase: Alveolar, Bilabial, Dental, etc.)
        for letter_dir in self.classes:
            letter_path = os.path.join(self.root_dir, letter_dir)

            for syllable_dir in os.listdir(letter_path):  # Subcarpetas de cada clase, como L, N, etc.
                syllable_path = os.path.join(letter_path, syllable_dir)

                for subsyllable_dir in os.listdir(syllable_path):  # Subcarpetas como LA, LE, LI, etc.
                    subsyllable_path = os.path.join(syllable_path, subsyllable_dir)
                    sample_paths = [os.path.join(subsyllable_path, sample_dir) for sample_dir in os.listdir(subsyllable_path)]
                    random.shuffle(sample_paths)  # Mezcla las muestras de forma aleatoria

                    # Asegurarse de que haya suficientes carpetas para train, val, y test
                    if len(sample_paths) >= 68:  # Asegurando que hay al menos 68 muestras
                        # Divide las muestras en train, val, y test
                        train_samples = sample_paths[:50]
                        val_samples = sample_paths[50:59]
                        test_samples = sample_paths[59:68]

                        for sample_dir in train_samples:
                            frame_files = [os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if len(frame_files) == self.frames_per_sample:
                                all_files[letter_dir].append(('train', frame_files))  # Marcar como 'train'

                        for sample_dir in val_samples:
                            frame_files = [os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if len(frame_files) == self.frames_per_sample:
                                all_files[letter_dir].append(('validation', frame_files))  # Marcar como 'validation'

                        for sample_dir in test_samples:
                            frame_files = [os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if len(frame_files) == self.frames_per_sample:
                                all_files[letter_dir].append(('test', frame_files))  # Marcar como 'test'

        # Organiza las muestras
        train, val, test = [], [], []
        for letter, samples in all_files.items():
            for sample in samples:
                mode, frame_files = sample
                if mode == 'train':
                    train.append((letter, frame_files))
                elif mode == 'validation':
                    val.append((letter, frame_files))
                elif mode == 'test':
                    test.append((letter, frame_files))

        return {'train': train, 'validation': val, 'test': test}

    def __getitem__(self, index):
        """
        Genera lotes de datos.

        Parámetros:
        - index: Índice del lote a generar.

        Retorna:
        - X_batch: Lote de imágenes.
        - y_batch: Lote de etiquetas (one-hot encoding).
        """
        batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = np.zeros((self.batch_size, self.frames_per_sample, *self.target_size, self.n_channels))
        y_batch = np.zeros((self.batch_size, len(self.classes)))

        # Procesa cada muestra del lote
        for i, (class_label, frames) in enumerate(batch_samples):
            class_index = self.class_to_index[class_label]

            # Carga los frames de la muestra
            images = [img_to_array(load_img(frame, color_mode='rgb', target_size=self.target_size)) / 255.0 for frame in frames]
            frames_array = np.array(images)

            X_batch[i] = frames_array
            y_batch[i, class_index] = 1   # Etiqueta one-hot

        return X_batch, y_batch

    def on_epoch_end(self):
        """
        Si shuffle está activado, mezcla las muestras al final de cada época.
        """
        if self.shuffle:
            self.samples = sk_shuffle(self.samples)

## FUNCIONES EXTRA (OPCIONALES) ##
#Funciones para observar los datos dentro de los generadores.

# Función para imprimir las muestras de cada partición (train, validation, test):
def print_sample_directories(generator):
    # Imprime las muestras de entrenamiento (train)
    print("Muestras en TRAIN:")
    for letter, frames in generator.dataset_info['train']:
        print(f"Clase: {letter}, Directorio de muestra: {os.path.dirname(frames[0])}")

    # Imprime las muestras de validación (validation)
    print("\nMuestras en VALIDATION:")
    for letter, frames in generator.dataset_info['validation']:
        print(f"Clase: {letter}, Directorio de muestra: {os.path.dirname(frames[0])}")

    # Imprime las muestras de prueba (test)
    print("\nMuestras en TEST:")
    for letter, frames in generator.dataset_info['test']:
        print(f"Clase: {letter}, Directorio de muestra: {os.path.dirname(frames[0])}")

# Esta función cuenta cuántas muestras hay en cada una de las particiones del dataset
def count_samples_per_division(generator):
    # Cuenta las muestras en el conjunto de entrenamiento (train)
    train_samples = len(generator.dataset_info['train'])

    # Cuenta las muestras en el conjunto de validación (validation)
    validation_samples = len(generator.dataset_info['validation'])

    # Cuenta las muestras en el conjunto de prueba (test)
    test_samples = len(generator.dataset_info['test'])

    # Imprime el total de muestras en cada partición
    print("\n\n")
    print(f"Total de muestras en TRAIN: {train_samples}")
    print(f"Total de muestras en VALIDATION: {validation_samples}")
    print(f"Total de muestras en TEST: {test_samples}")

# Esta función toma un batch de secuencias generadas por el generador y las visualiza.
# Por defecto, visualiza 4 secuencias, pero se puede ajustar el número de secuencias a mostrar.
def visualize_generated_sequences(generator, num_sequences=4):
    # Obtiene el primer batch del generador
    X_batch, y_batch = generator.__getitem__(0)

    # Asegura que no se intenten visualizar más secuencias de las que hay en el batch
    num_sequences = min(num_sequences, X_batch.shape[0])

    # Visualiza las secuencias
    for i in range(num_sequences):
        plt.figure(figsize=(15, 3))  # Ajusta el tamaño de la figura
        for j in range(generator.frames_per_sample):  # Recorre los frames de cada secuencia
            plt.subplot(1, generator.frames_per_sample, j + 1)  # Crea subplots para cada frame
            if generator.n_channels == 1:
                # Si las imágenes están en escala de grises
                plt.imshow(X_batch[i, j, :, :, 0], cmap='gray')
            else:
                # Si las imágenes están en formato RGB
                plt.imshow(X_batch[i, j, :, :, :])
            plt.axis('off')  # Oculta los ejes
        plt.show()  # Muestra las imágenes



# Código para verificar las muestras entregadas por los generadores
def contar_muestras_generador(generator):
    contador = Counter()
    for i in range(len(generator)):
        _, y_batch = generator[i]
        etiquetas = np.argmax(y_batch, axis=1)  # Obtener los índices de las clases
        for etiqueta in etiquetas:
            contador[etiqueta] += 1
    return contador

def print_sample_count(test_generator, sample_count):
    # Mapear índices a nombres de clases
    indice_a_clase = {v: k for k, v in test_generator.class_to_index.items()}

    # Imprimir el conteo de muestras por clase
    for idx, count in sample_count.items():
        print(f"Clase: {indice_a_clase[idx]}, Muestras: {count}")

## ---------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------- ##

# FUNCIÓN PARA IMPRIMIR LAS GRÁFICAS DE PRECISIÓN Y PERDIDA
def print_accuracy_loss(history):
    # Sumarizar la historia de la precisión
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    plt.show()

    # Sumarizar la historia de la pérdida
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    plt.show()


# FUNCIÓN PARA IMPRIMIR LA MATRIZ DE CONFUSIÓN Y EL REPORTE DE CLASIFICACIÓN
def print_conf_matriz_classif_report(model, test_generator, file_name):
    # Matriz de confusión e Informe de clasificación
    y_true = []
    y_pred = []

    # Recorremos el conjunto de datos de test para obtener las etiquetas reales y las predicciones
    for i in range(len(test_generator)):
        X_test, y_test = test_generator.__getitem__(i)
        preds = model.predict(X_test)

        y_true.extend(np.argmax(y_test, axis=1))  # Etiquetas reales
        y_pred.extend(np.argmax(preds, axis=1))   # Predicciones

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_to_index.keys(), yticklabels=test_generator.class_to_index.keys())
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas verdaderas')
    plt.title('Matriz de Confusión')
    # plt.show()

    plt.savefig(f'{directory_name}/matriz_confusion.png')

    # Informe de clasificación

    cf_report = classification_report(y_true, y_pred, target_names=test_generator.class_to_index.keys())
    # print(classification_report(y_true, y_pred, target_names=test_generator.class_to_index.keys()))

    with open(f"{directory_name}/reporte_clasificacion.txt", "w") as archivo:
      archivo.write(cf_report)

    accuracy_por_clase = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    with open(f"{directory_name}/accuracy_por_clase.txt", "w") as archivo:
      for idx, accuracy in enumerate(accuracy_por_clase):
          archivo.write(f"{list(test_generator.class_to_index.keys())[idx]}: {accuracy:.2f}\n")

def save_list(file_name, lista):
  with open(file_name, "w") as archivo:
    for elemento in lista:
        archivo.write(f'{elemento:.5f}'+ "\n")

def custom_CNN():
    inputs = Input(shape=(128, 128, 3))

    x = Conv2D(n_filters, (3, 3), activation='relu')(inputs)
    x = Conv2D(n_filters, (3, 3), activation='relu')(x)
    x = Conv2D(n_filters, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)


    x = Conv2D(n_filters, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(n_filters, (3, 3), activation='relu')(x)
    x = Conv2D(n_filters, (3, 3), activation='relu')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    outputs = Dense(512, activation='relu')(x)

    return Model(inputs, outputs)

def CNN_LSTM(input_shape, num_classes, lstm_units=256):

    inputs = Input(shape = input_shape)

    custom_cnn = custom_CNN()

    cnn_out = TimeDistributed(custom_cnn)(inputs)

    lstm = LSTM(lstm_units, return_sequences=False)
    lstm_out = lstm(cnn_out)

    outputs = Dense(num_classes, activation='softmax')(lstm_out)

    # Crear y compilar el modelo
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
  epochs = 100
  batch_size = 64
  learning_rate = 0.0001

  model_name = 'punto_articulacion'

  num_classes = 6

  dataset_path = 'Dataset_TT/Punto de articulacion'

  n_frames_per_sample = 18  # Número de frames por muestra, ya que cada video se ha dividido en 18 imágenes
  frame_height = 128   # Altura de cada frame (en píxeles)
  frame_width = 128    # Ancho de cada frame (en píxeles)
  n_filters = 32      # Número de filtros para las capas convolucionales
  n_channels = 3       # Número de canales en las imágenes (3 para RGB)
  model_name = model_name + '_LRCN_2'

  directory_name =f'{model_name}_b_{batch_size}_e_{learning_rate}_n_{n_filters}'
  
  try:
    os.mkdir(directory_name)
  except:
    print('Ya existe el directorio')

  train_generator = CustomDataGenerator(dataset_path, mode='train', batch_size=batch_size, shuffle=True)
  validation_generator = CustomDataGenerator(dataset_path, mode='validation', batch_size=batch_size, shuffle=False)
  test_generator = CustomDataGenerator(dataset_path, mode='test', batch_size=batch_size, shuffle=False)
  
  # print_sample_directories(train_generator)
  count_samples_per_division(train_generator)
  # visualize_generated_sequences(train_generator, num_sequences=5)

  conteo_muestras = contar_muestras_generador(test_generator)
  # print_sample_count(test_generator, conteo_muestras)

  input_shape = (n_frames_per_sample, frame_height, frame_width, n_channels)
  model = CNN_LSTM(input_shape, num_classes = num_classes)

  checkpoint_filepath = f'{directory_name}/{model_name}_best_weigths.keras'
  
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath= checkpoint_filepath,   # Nombre del archivo para guardar los mejores pesos
    monitor='val_accuracy',            # Métrica a monitorear
    save_best_only=True,           # Guardar solo si es el mejor
    mode='max',                    # Modo "min" para guardar el menor valor (útil para "val_loss")
    verbose=1                      # Muestra mensajes cuando guarda los pesos
  )

  early_stopping = EarlyStopping(monitor="val_accuracy", patience=50 , start_from_epoch=0, restore_best_weights=True)
#   model.summary()
  
  history = model.fit(
    train_generator,  # Pasa el generador directamente para entrenamiento
    validation_data=validation_generator,  # Usa el generador para validación
    epochs=epochs,  # Número de épocas
    # steps_per_epoch=len(train_generator),
    # validation_steps=len(validation_generator)
    callbacks = [early_stopping]
  )

  model.save(f'{directory_name}/{model_name}.keras')

  accuracy_train_model = history.history['accuracy']
  save_list(f'{directory_name}/{model_name}_accuracy.txt', accuracy_train_model)

  accuracy_val_model = history.history['val_accuracy']
  save_list(f'{directory_name}/{model_name}_val_accuracy.txt', accuracy_val_model)

  loss_train_model = history.history['loss']
  save_list(f'{directory_name}/{model_name}_loss.txt', loss_train_model)

  loss_val_model = history.history['val_loss']
  save_list(f'{directory_name}/{model_name}_val_loss.txt', loss_val_model)

  model = tf.keras.models.load_model(checkpoint_filepath)

  print_conf_matriz_classif_report(model=model, test_generator=test_generator, file_name=directory_name)