# model_training.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

def train_model(X_train, Y_train, X_val, Y_val, input_shape, epochs=30):  # `epochs` als Parameter hinzufügen
    model = Sequential([
        Conv2D(32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=input_shape),
        Conv2D(32, kernel_size=(5,5), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu'),
        Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, Y_train, batch_size=86), 
                        epochs=epochs,
                        validation_data=(X_val, Y_val),
                        verbose=2,
                        steps_per_epoch=X_train.shape[0] // 86,
                        callbacks=[learning_rate_reduction])

    return model, history
