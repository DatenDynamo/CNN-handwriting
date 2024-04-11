from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Annahme: Modelle sind geladen als model_drop_full, model_drop_move, model_drop, model2
# Annahme: Daten sind geladen und vorbereitet als x_train, y_train, x_train_all, y_train_all

# Definiere ein EarlyStopping Callback, das für jedes Modell verwendet werden kann
early_stopping_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            verbose=1,
            mode="max",
            restore_best_weights=True)

# Liste der Modelle und ihrer spezifischen Trainingsparameter
models_with_params = [
    (model_drop_full, x_train_all, y_train_all, 64, 150),
    (model_drop_move, x_train_all, y_train_all, 64, 150),
    (model_drop, x_train, y_train, 64, 150),
    (model2, x_train, y_train, 64, 150),
    (model, x_train, y_train, 64, 150)
]

histories = []

# Trainiere jedes Modell mit seinen spezifischen Parametern
for model, x_data, y_data, batch_size, epochs in models_with_params:
    history = model.fit(x_data, y_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[early_stopping_callback])
    histories.append(history)

plt.figure(figsize=(14, 5))

for model_name, history in histories:
    plt.subplot(1, 2, 1)
    plt.semilogy(history.history['loss'], label=f'{model_name} Training Loss')
    plt.semilogy(history.history['val_loss'], label=f'{model_name} Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

plt.subplot(1, 2, 1)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.style.use('ggplot')

plt.show()