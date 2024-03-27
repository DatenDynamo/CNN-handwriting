# _05_model_plot.py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history, accuracy_ylim_bottom=None, accuracy_ylim_top=None, save_plot=False):
    """Visualisiert den Verlauf von Loss und Accuracy während des Trainings und der Validierung über die Epochen hinweg."""
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
    plt.style.use('ggplot')
    # Setzen der y-Achsen-Grenzen, falls spezifiziert
    if accuracy_ylim_bottom is not None and accuracy_ylim_top is not None:
        plt.ylim(accuracy_ylim_bottom, accuracy_ylim_top)

    if save_plot:
        plt.savefig("images/best.png")  # Speichert den Plot als PNG-Datei
        plt.close()

    plt.show()


# Konfusionmatrix


def plot_konfusionsmatrix(Y_test, Y_pred, is_better, klassen=range(10), titel='Konfusionsmatrix'):
    """
    Diese Funktion plottet die Konfusionsmatrix.

    Args:
    - Y_test: Die wahren Klassen (labels) des Testdatensatzes (one hot encoded).
    - Y_pred: Die vom Modell vorhergesagten Klassen (als Wahrscheinlichkeiten).
    - ist_besser: Eine Boolesche Variable, die angibt, ob das aktuelle Modell besser ist als vorherige Modelle.
    - klassen: Die Liste der Klassen, die im Datensatz vorkommen. Standardmäßig von 0 bis 9 für den MNIST-Datensatz.
    - titel: Der Titel der Konfusionsmatrix.
    """
    # Vorhersagen in Klassen umwandeln
    Y_pred_klassen = np.argmax(Y_pred, axis=1)
    # Wahre Klassen aus one hot Vektoren umwandeln
    Y_wahr = np.argmax(Y_test, axis=1)
    # Konfusionsmatrix berechnen
    konfusionsmatrix = confusion_matrix(Y_wahr, Y_pred_klassen)
    # Konfusionsmatrix plotten
    plt.figure(figsize=(10, 8))
    sns.heatmap(konfusionsmatrix, annot=True, fmt='d', cmap='Blues')
    plt.tight_layout()
    plt.title(titel)
    plt.ylabel('Wahre Ziffer')
    plt.xlabel('Durch Modell bestimmte Ziffer')
    plt.style.use('ggplot')
    # Wenn das aktuelle Modell besser ist, speichere die Konfusionsmatrix
    if is_better:
        plt.savefig("images/beste_konfusionsmatrix.png")
    plt.show()


# Display Errors

def fehler_bestimmen(Y_test, Y_pred, test):
    """Identifies and shows the most significant errors in the model's predictions."""
    Y_pred_klassen = np.argmax(Y_pred, axis=1)
    Y_wahr = np.argmax(Y_test, axis=1)
    fehler = (Y_pred_klassen - Y_wahr != 0)

    Y_pred_klassen_fehler = Y_pred_klassen[fehler]
    Y_pred_fehler = Y_pred[fehler]
    Y_wahr_fehler = Y_wahr[fehler]
    test_daten_fehler = test[fehler]

    # Probabilities of the incorrectly predicted numbers
    Y_pred_fehler_wahrscheinlichkeit = np.max(Y_pred_fehler, axis=1)

    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_fehler, Y_wahr_fehler, axis=1))

    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_fehler_wahrscheinlichkeit - true_prob_errors

    # Sorted list of the delta prob errors
    sorted_delta_errors = np.argsort(delta_pred_true_errors)

    # Most significant errors
    wichtigste_fehler = sorted_delta_errors[-9:]

    return wichtigste_fehler, test_daten_fehler, Y_pred_klassen_fehler, Y_wahr_fehler
   


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows images with their predicted and real labels for the provided indices."""
    n = 0
    ncols = 3
    nrows = 3
    
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12, 12))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)), cmap='gray_r') # Farben invertieren mit cmap='gray_r'
            ax[row, col].set_title("Durch Modell bestimmte Ziffer: {}\nWahre Ziffer: {}".format(pred_errors[error], obs_errors[error]))
            ax[row, col].axis('off')
            n += 1
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()


def plot_activation_maps(model, layer_name, input_image):
    """Zeigt die Aktivierungskarten/Feature-Maps einer spezifischen Schicht eines CNN für ein gegebenes Eingabebild."""
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_model.predict(input_image)

    plt.figure(figsize=(20, 20))
    for i, activation_map in enumerate(intermediate_output[0]):
        plt.subplot(6, 6, i+1)
        plt.imshow(activation_map, cmap='viridis')
        plt.axis('off')
    plt.style.use('ggplot')
    plt.show()

def plot_filters(layer):
    """Visualisiert die gelernten Filter einer spezifischen Convolutional-Layer eines CNN."""

    filters, biases = layer.get_weights()
    plt.figure(figsize=(10, 10))
    for i, filter in enumerate(filters[:, :, :, 0]):  
        plt.subplot(6, 6, i+1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
    plt.style.use('ggplot')
    plt.show()


def visualize_embeddings(model, layer_name, input_data, labels, num_samples=1000, method='tsne'):
    """
    Reduziert die Dimensionalität der Feature-Embeddings einer 
    Modellschicht und visualisiert sie mit T-SNE oder PCA.

    - model: Das trainiertes Keras Modell.
    - layer_name: Der Name der Schicht, deren Ausgaben visualisiert werden sollen.
    - input_data: Die Eingabedaten für das Modell.
    - labels: Die wahren Labels der Eingabedaten.
    - num_samples: Anzahl der Datenpunkte, die für die Visualisierung verwendet werden sollen.
    - method: 'tsne' oder 'pca' zur Bestimmung der Reduktionsmethode.
    """
    np.random.seed(0)  # Für reproduzierbare Ergebnisse
    
    # Stichprobe der Eingabedaten und Labels für die Visualisierung
    indices = np.random.choice(range(len(input_data)), num_samples, replace=False)
    sampled_data = input_data[indices]
    sampled_labels = labels[indices]
    
    # Erstellen eines Modells, das die Ausgaben der spezifischen Schicht extrahiert
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Extrahieren der Features
    intermediate_output = intermediate_model.predict(sampled_data)
    
    # Reduzieren der Dimensionalität mit T-SNE oder PCA
    if method == 'tsne':
        embeddings = TSNE(n_components=2).fit_transform(intermediate_output)
    else:  # Standardmäßig verwenden wir PCA, wenn 'tsne' nicht ausdrücklich angegeben ist
        embeddings = PCA(n_components=2).fit_transform(intermediate_output)
    
    # Visualisieren der reduzierten Daten
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=sampled_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(f'{method.upper()} visualized embeddings from {layer_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.style.use('ggplot')
    plt.show()