# _05_model_plot.py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import Model

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

    # Setzen der y-Achsen-Grenzen, falls spezifiziert
    if accuracy_ylim_bottom is not None and accuracy_ylim_top is not None:
        plt.ylim(accuracy_ylim_bottom, accuracy_ylim_top)

    if save_plot:
        plt.savefig("images/best.png")  # Speichert den Plot als PNG-Datei
        plt.close()

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
    plt.show()

def plot_filters(layer):
    """Visualisiert die gelernten Filter einer spezifischen Convolutional-Layer eines CNN."""

    filters, biases = layer.get_weights()
    plt.figure(figsize=(10, 10))
    for i, filter in enumerate(filters[:, :, :, 0]):  
        plt.subplot(6, 6, i+1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
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
    plt.show()