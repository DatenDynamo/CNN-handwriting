# model_training.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_mnist_data
from data_preprocess import preprocess_data

# tf.config.list_physical_devices('GPU')

# Recreate the exact same model, including its weights and the optimizer
def create_functional_model():
    inputs = keras.Input(shape=(28, 28, 1), name="conv2d_input")
    
    # Convolutional layers
    x = layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d")(inputs)
    x = layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d")(x)
    x = layers.Dropout(0.25, name="dropout")(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_2")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_1")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)
    
    # Flattening the 3D output to 1D
    x = layers.Flatten(name="flatten")(x)
    
    # Dense layers for classification
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="dense_1")(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="sequential")
    
    return model



pretrained_model = create_functional_model()
pretrained_model.load_weights('Model 99,72/model_weights.h5')

extracted_layers = pretrained_model.layers[:-1]

model = keras.Sequential(extracted_layers)
model.summary()

optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

TEST_SIZE = 0.2 # Größe von den Validierungsdaten beim Split // Achtung: zum Testen des Modells wird IMMER der MNIST-Testdatensatz verwendet, siehe: https://ai.stackexchange.com/questions/37577/how-is-mnist-only-providing-the-training-and-the-test-sets-what-about-the-valid
epochs = 150 # Anzahl der Epochen // bricht aber sowieso nach der "idealen" Anzahl ab wenn early_stopping_enabled TRUE ist
batch_size = 128
SEED = 2
early_stopping_enabled = True


    # Daten laden
train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Daten vorverarbeiten, mit dem global festgelegten Test Size
X_train, X_val, Y_train, Y_val, test, Y_test, input_shape = preprocess_data(train_images, train_labels, test_images, test_labels, test_size=TEST_SIZE, random_seed=SEED)

score = model.evaluate(test, Y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')