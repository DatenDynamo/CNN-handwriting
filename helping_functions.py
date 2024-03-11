import numpy as np
import matplotlib.pyplot as plt

def plot_samples(xdata, ydata, amount=40):
    for number in range(10):
        indices = np.random.choice(np.where(ydata == number)[0], size=amount, replace=False)
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle("Examples for number " + str(number) + ":")
        nplt = 1
        for index in indices:
            ax = fig.add_subplot(4, 10, nplt)
            ax.axis('off')
            ax.imshow(xdata[index], cmap='gray')  # Verwenden Sie ax.imshow mit dem Parameter cmap='gray'
            nplt += 1
        plt.show()  # Stellen Sie sicher, dass plt.show() aufgerufen wird, um das Diagramm anzuzeigen
