# _02_data_preprocess.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

def preprocess_data(train_images, train_labels, test_images, test_labels, val_size=0.1, random_seed=2):
    if K.image_data_format() == 'channels_first':
        train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
        test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
    
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=val_size, random_state=random_seed)

    return X_train, X_val, Y_train, Y_val, test_images, test_labels, input_shape
