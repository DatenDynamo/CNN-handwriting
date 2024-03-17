#Grafische Visaliserung von Daten#
#https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy#
#https://github.com/as3eem/MNIST-Data-Analysis/blob/master/MNIST.ipynb#

import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(train_labels, test_labels):
    plt.figure(figsize=(8, 6))
    plt.hist([train_labels, test_labels], bins=np.arange(11)-0.5, rwidth=0.8, label=['Train', 'Test'])
    plt.xticks(np.arange(10))
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Beispiele')
    plt.title('Verteilung Klassen im MNIST-Datensatz')
    plt.legend()
    plt.show()

def plot_sample_images(train_images, train_labels):
    num_rows = 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        rand_idx = np.random.randint(len(train_images))
        ax.imshow(train_images[rand_idx], cmap='gray')
        ax.set_title("Label: {}".format(train_labels[rand_idx]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_pixel_intensity_distribution(train_images, train_labels):
    class_labels = np.unique(train_labels)
    plt.figure(figsize=(10, 8))
    for label in class_labels:
        plt.hist(train_images[train_labels == label].flatten(), bins=50, alpha=0.5, label=str(label))
    plt.xlabel('Pixelintensität')
    plt.ylabel('Häufigkeit')
    plt.title('Verteilung Pixelintensitäten nach Klassen')
    plt.legend()
    plt.show()

def plot_correlation_matrix(train_images):
    correlation_matrix = np.corrcoef(train_images.reshape(train_images.shape[0], -1), rowvar=False)
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Korrelationskoeffizient')
    plt.title('Korrelationsmatrix der Pixelintensitäten')
    plt.show()

plot_class_distribution(train_labels, test_labels)

plot_sample_images(train_images, train_labels)

plot_pixel_intensity_distribution(train_images, train_labels)

plot_correlation_matrix(train_images)
