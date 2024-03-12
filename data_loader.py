# data_loader.py
from tensorflow.keras.datasets import mnist
import json

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels

def save_best_parameters(test_loss, test_accuracy, parameters, file_path='best_parameters.json'):
    try:
        # Versuchen, die bisher besten Parameter zu laden
        with open(file_path, 'r') as file:
            best_parameters = json.load(file)
            best_test_loss = best_parameters.get('test_loss', float('inf'))
            best_test_accuracy = best_parameters.get('test_accuracy', 0)
    except FileNotFoundError:
        best_test_loss = float('inf')
        best_test_accuracy = 0

    # Überprüfen, ob die aktuellen Metriken besser sind
    if test_loss < best_test_loss or test_accuracy > best_test_accuracy:
        # Aktuelle Parameter und Metriken als die neuen besten speichern
        best_parameters = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'parameters': parameters
        }
        with open(file_path, 'w') as file:
            json.dump(best_parameters, file, indent=4)
        print("Neue beste Parameter und Metriken gespeichert.")
    else:
        print("Die aktuellen Metriken sind nicht besser als die bisher besten.")

def load_best_parameters(file_path='best_parameters.json'):
    try:
        with open(file_path, 'r') as file:
            best_parameters = json.load(file)
            return best_parameters['parameters']
    except FileNotFoundError:
        return None  # oder setzen Sie Standardwerte