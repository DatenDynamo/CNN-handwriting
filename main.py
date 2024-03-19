# main.py
from data_loader import load_mnist_data
from data_preprocess import preprocess_data
from model_training import train_model
from my_plot import plot_training_history
from data_loader import load_best_parameters
from data_loader import save_best_parameters

# Global festgelegte Parameter
TEST_SIZE = 0.2 # Größe von den Validierungsdaten beim Split // Achtung: zum Testen des Modells wird IMMER der MNIST-Testdatensatz verwendet, siehe: https://ai.stackexchange.com/questions/37577/how-is-mnist-only-providing-the-training-and-the-test-sets-what-about-the-valid
epochs = 150 # Anzahl der Epochen // bricht aber sowieso nach der "idealen" Anzahl ab wenn early_stopping_enabled 1 ist
batch_size = 128
SEED = 2
early_stopping_enabled = 1

def main():
    # Daten laden
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Daten vorverarbeiten, mit dem global festgelegten Test Size
    X_train, X_val, Y_train, Y_val, test, Y_test, input_shape = preprocess_data(train_images, train_labels, test_images, test_labels, test_size=TEST_SIZE, random_seed=SEED)
    
    best_parameters = load_best_parameters()
    if best_parameters:
        # Verwende die besten Parameter für Training
        print("Beste Parameter geladen:", best_parameters)
    else:
        print("Keine gespeicherten besten Parameter gefunden. Verwenden der Standardwerte.")

    # Modell trainieren, mit der global festgelegten Anzahl von Epochen
    model, history, callbacks, early_stopping_callback = train_model(X_train, Y_train, X_val, Y_val, input_shape, epochs=epochs, batch_size=batch_size, early_stopping_enabled=early_stopping_enabled)
    
    model.summary()
    # Modell evaluieren
    score = model.evaluate(test, Y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    
    # Speichern der Parameter für VErgleich bei zukünftigen Training
    if early_stopping_callback:
        tatsaechlich_benoetigte_epochs = early_stopping_callback.stopped_epoch + 1
    else:
        tatsaechlich_benoetigte_epochs = epochs
    test_loss = score[0]
    test_accuracy = score[1]
    parameters = {
        'angegebene epochs': epochs,
        'tatsaelich benoetigte epochs': tatsaechlich_benoetigte_epochs,
        'batch_size': batch_size,
        'Split der Validierungsdaten:': TEST_SIZE,
        'Seed' : SEED,
    }

    is_better = save_best_parameters(test_loss, test_accuracy, parameters)
    
    if is_better:
        json_string = model.to_json()
        open('model_architecture.json','w').write(json_string)
        # Save the weights
        model.save_weights('model_weights.h5',overwrite=True)

    # Trainingsverlauf plotten
    plot_training_history(history)
    plot_training_history(history, accuracy_ylim_bottom=0.97, accuracy_ylim_top=1.0)

if __name__ == '__main__':
    main()
