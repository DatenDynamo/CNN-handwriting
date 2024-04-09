# _00_main.py
from pickle import TRUE
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn

from _01_data_loader import load_mnist_data, load_best_parameters
from _02_data_preprocess import preprocess_data
from _03_model_training import train_model
from _04_data_saver import save_best_parameters, update_readme_from_json
from _05_model_plot import plot_training_history, plot_activation_maps, plot_filters, visualize_embeddings, plot_konfusionsmatrix, fehler_bestimmen, display_errors

# Global festgelegte Parameter
VAL_SIZE = 0.1 # Größe von den Validierungsdaten beim Split // Achtung: zum Testen des Modells wird IMMER der MNIST-Testdatensatz verwendet, siehe: https://ai.stackexchange.com/questions/37577/how-is-mnist-only-providing-the-training-and-the-test-sets-what-about-the-valid
epochs = 2 # Anzahl der Epochen // bricht aber sowieso nach der "idealen" Anzahl ab wenn early_stopping_enabled TRUE ist
batch_size = 64
SEED = 2
early_stopping_enabled = True

def main():
    # Daten laden
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Daten vorverarbeiten, mit dem global festgelegten Split für die Validierungsdaten
    X_train, X_val, Y_train, Y_val, test, Y_test, input_shape = preprocess_data(train_images, train_labels, test_images, test_labels, val_size=VAL_SIZE, random_seed=SEED)
    

    # Modell trainieren, mit der global festgelegten Anzahl von Epochen
    model = load_model("model.h5")
    
    model.summary()
    # Modell evaluieren
    score = model.evaluate(test, Y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
    

    is_better = True


    # Konfusionsmatrix
    Y_pred = model.predict(test)
    plot_konfusionsmatrix(Y_test, Y_pred, is_better, klassen=range(10), titel='Konfusionsmatrix', save_plot=is_better)

    # Display some error results 
    wichtigste_fehler, test_daten_fehler, Y_pred_klassen_fehler, Y_wahr_fehler, delta_pred_true_errors = fehler_bestimmen(Y_test, Y_pred, test)
    display_errors(wichtigste_fehler, test_daten_fehler, Y_pred_klassen_fehler, Y_wahr_fehler, delta_pred_true_errors, save_plot=is_better)
    

    if is_better:
        return is_better
    
    plot_activation_maps(model, 'conv2d_1', test[0:1])

    plot_filters(model.layers[0]) 

    visualize_embeddings(model, 'dense_1', test, test_labels)  # Angenommen, 'dense_1' ist eine Ihrer tiefen Schichten


if __name__ == '__main__':
    main()

