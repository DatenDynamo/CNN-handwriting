# my_plot.py
import matplotlib.pyplot as plt

def plot_training_history(history, accuracy_ylim_bottom=None, accuracy_ylim_top=None):
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

    # Setzen der y-Achsen-Grenzen, falls spezifiziert
    if accuracy_ylim_bottom is not None and accuracy_ylim_top is not None:
        plt.ylim(accuracy_ylim_bottom, accuracy_ylim_top)

    plt.show()
