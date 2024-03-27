# _00_main.py
from pickle import TRUE
from _01_data_loader import load_mnist_data, load_best_parameters
from _02_data_preprocess import preprocess_data
from _03_model_training import train_model
from _04_data_saver import save_best_parameters, update_readme_from_json
from _05_model_plot import plot_training_history, plot_activation_maps, plot_filters, visualize_embeddings, plot_konfusionsmatrix, fehler_bestimmen, display_errors
# Global festgelegte Parameter
VAL_SIZE = 0.1 # Größe von den Validierungsdaten beim Split // Achtung: zum Testen des Modells wird IMMER der MNIST-Testdatensatz verwendet, siehe: https://ai.stackexchange.com/questions/37577/how-is-mnist-only-providing-the-training-and-the-test-sets-what-about-the-valid
epochs = 150 # Anzahl der Epochen // bricht aber sowieso nach der "idealen" Anzahl ab wenn early_stopping_enabled TRUE ist
batch_size = 64
SEED = 2
early_stopping_enabled = True

def main():
    # Daten laden
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Daten vorverarbeiten, mit dem global festgelegten Split für die Validierungsdaten
    X_train, X_val, Y_train, Y_val, test, Y_test, input_shape = preprocess_data(train_images, train_labels, test_images, test_labels, val_size=VAL_SIZE, random_seed=SEED)
    
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
        'Split der Validierungsdaten:': VAL_SIZE,
        'Seed' : SEED,
    }

    is_better = save_best_parameters(test_loss, test_accuracy, parameters)
    
    if is_better:
        json_string = model.to_json()
        open('model_architecture.json','w').write(json_string)
        
        # Update readme
        json_file_path = 'best_parameters.json'
        readme_file_path = 'README.md'
        start_marker = '<!-- START -->\n'
        end_marker = '<!-- END -->\n'
        update_readme_from_json(json_file_path, readme_file_path, start_marker, end_marker)


        # Save the weights
        model.save('model.h5',overwrite=True)

    # Trainingsverlauf plotten
    if is_better:
        plot_training_history(history, save_plot=True)
    else:
        plot_training_history(history, save_plot=False)

    plot_training_history(history, accuracy_ylim_bottom=0.97, accuracy_ylim_top=1.0)

    # Konfusionsmatrix
    Y_pred = model.predict(test)
    plot_konfusionsmatrix(Y_test, Y_pred, is_better, klassen=range(10), titel='Konfusionsmatrix')

    # Display some error results 
    wichtigste_fehler, test_daten_fehler, Y_pred_klassen_fehler, Y_wahr_fehler = fehler_bestimmen(Y_test, Y_pred, test)
    display_errors(wichtigste_fehler, test_daten_fehler, Y_pred_klassen_fehler, Y_wahr_fehler)
    

    if is_better:
        return is_better
    
    plot_activation_maps(model, 'conv2d_1', test[0:1])

    plot_filters(model.layers[0]) 

    visualize_embeddings(model, 'dense_1', test, test_labels)  # Angenommen, 'dense_1' ist eine Ihrer tiefen Schichten


if __name__ == '__main__':
    main()



