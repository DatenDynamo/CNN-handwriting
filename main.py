from helping_functions import (
    plot_samples,
)


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

# Grundlegende Parameter für die Ausführung
TFliteNamingAndVersion = "dig1410s3"   # Wird für den tflite-Dateinamen verwendet
Epoch_Anz = 100 # 500                        # Anzahl der Epochen

# Laden des MNIST-Datensatzes
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisierung der Daten
x_train_norm, x_test_norm = x_train / 255, x_test / 255

print('x_train:', x_train_norm.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test_norm.shape)
print('y_test:', y_test.shape)



# Hinzufügen einer Dimension, um den Bildern einen einzelnen Kanal hinzuzufügen
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Konvertierung in One-Hot-Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Modellarchitektur definieren
inputs = Input(shape=(28, 28, 1))
x = BatchNormalization()(inputs)
x = Conv2D(32, (3, 3), padding='same', activation="relu")(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(64, (3, 3), padding='same', activation="relu")(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)

# Modell kompilieren
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modellzusammenfassung anzeigen
model.summary()

# Training des Modells
history = model.fit(x_train, y_train, epochs=Epoch_Anz, validation_data=(x_test, y_test), batch_size=32)

# Visualisierung des Trainingsverlaufs
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.semilogy(history.history['loss'], label='Training Loss')
plt.semilogy(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()
