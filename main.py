# main.py
from data_loader import load_mnist_data
from data_preprocess import preprocess_data
from model_training import train_model
from my_plot import plot_training_history

# Global festgelegte Parameter
TEST_SIZE = 0.1
EPOCHS = 100

def main():
    # Daten laden
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Daten vorverarbeiten, mit dem global festgelegten Test Size
    X_train, X_val, Y_train, Y_val, test, Y_test, input_shape = preprocess_data(train_images, train_labels, test_images, test_labels, test_size=TEST_SIZE)
    
    # Modell trainieren, mit der global festgelegten Anzahl von Epochen
    model, history = train_model(X_train, Y_train, X_val, Y_val, input_shape, epochs=EPOCHS)
    
    # Modell evaluieren
    score = model.evaluate(test, Y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    
    # Trainingsverlauf plotten
    plot_training_history(history)

if __name__ == '__main__':
    main()
