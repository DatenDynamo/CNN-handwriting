# CNN für den MNIST-Datensatz (Gruppennumer: 101)


## Kurzbeschreibung
**Autor:** *NF*

In diesem GitLab-Projekt wollen wir gemeinsam am Code arbeiten, um das Modul Künstliche Intelligenz I im Wintersemester 2023/24 abzuschließen. Ziel des Projekts ist die Bearbeitung der folgenden Aufgabe:
>"Betrachten Sie den MNIST-Datensatz zur Handschriftenerkennung von Ziffern, wie in der Vorlesung betrachtet. Entwickeln Sie ein convolutional neural network (CNN) und vergleichen Sie Ihr Modell mit den in der Vorlesung behandelten Netzwerken."

![Bild von Training](images/best.png)

### Aktueller Trainingshighscore:
<!-- START -->
**test_loss:** 0.008994079194962978

**test_accuracy:** 0.996999979019165

**parameters:** {'angegebene epochs': 150, 'tatsaelich benoetigte epochs': 41, 'batch_size': 64, 'Split der Validierungsdaten:': 0.1, 'Seed': 2}
<!-- END -->


Der aktuelle Highscore beim Training ist auch in der `best_parameters.json` zu finden.

**test_accuracy:** 0.9965999722480774

![Bild1 von Training](images/Figure_1.png)


![Bild1 von Training](images/Figure_2.png)

## Activation maps von erstem Conv
**Autor:** *MG*

![Bild1 von Training](images/Figure_3.png)

## Filter
**Autor:** *MG*

![Bild1 von Training](images/Figure_4.png)

## TSNE Plot
**Autor:** *MG*

![Bild1 von Training](images/Figure_5.png)


## Hintergrund MNIST Datensatz

**Autor:** *KK*

Der MNIST (Modified National Institute of Standards and Technology database) Datensatz ist ein zahlenbasierter Datensatz mit knapp 70.000 Daten. Inhalt der daten sind handschriftlich geschriebene Zahlen von 0 bis 9 in 28x28 Pixel-Bildern, die auf individuelle Art und Weise aufgeschrieben worden sind.

![KK Sample vom MNIST Datensatz](images/KK_MNIST_sample.png)
![Durchschnitt aller Bilder](images/KK_Durchschnittsbilder.png)

Von diesen 70.000 Daten sind 60.000 Trainingsdaten und 10.000 Testdaten. Die 60.000 Trainingsdaten sind, wie der Name schon verrät, da um das Modell zu trainieren. Die 10.000 Bilder in den Testdaten sind dafür da, um die trainierten Daten zu vergleichen.  Ziel ist es, mit Hilfe von neuronalen Netzen die Zahlen richtig zu erkennen und zu klassifizieren.  Mithilfe von diesem freizugänglichen Datensatz können Neuronale Netze erstellt werden, dadurch das maschinelle Lernen im Allgemeinen gefördert werden. Das Training und die Evaluierung von Algorithmen für die optische Zeichenerkennung (OCR) und das maschinelle Lernen verwendet.
Auch kann man diese Daten in verschiedensten Varianten visuell aufzeigen. In den folgenden Bildern wurden die Daten in beispielsweise Cluster, PCA und t-SNE Visualisierung, da diese unserer Meinung nach die besten sind. Mit dem Cluster sollen die Daten gruppiert werden und unterschiedlich zu anderen Gruppen (Clustern) gemacht werden.

![Cluster](images/KK_Cluster.png)

Desweiteren wurde eine Prinicpal Compotent Analysis (PCA) Darstellung erzeugt. Die PCA soll Date mit hohen bzw. vielen Dimensionen sollen damit verstädnlicher gemacht werden indem man die Dimemsionen reduziert. Dabei sollen viele Varioatienen beibehalten werden. PCA transformiert die Daten in einen neuen Merkmalsraum, der durch die Hauptkomponenten definiert wird. Diese Hauptkomponenten sind die linearen Kombinationen der ursprünglichen Merkmale, die die maximale Varianz im Datensatz erklären.

![PCA](images/KK_PCA.png)

Und als letztes wurde der Datensatz mithilfe der t-Distributed Stochastic Neighbor Embedding (t-SNE) Methode visualisiert. Auch diese Methode wird genutzt, um die hochdimensionalen Räume zu vereinfachen bzw. diese zu reduzieren. Aber im Gegensatz zum PCA Prinzip, wird hier versucht die Merkmale im gleichen Datenraum beizubehalten.

![tSNE](images/KK_tSNE.png)

**Quellen:**
- **[https://docs.ultralytics.com/de/datasets/classify/mnist](https://docs.ultralytics.com/de/datasets/classify/mnist)**
- **[https://www.tensorflow.org/datasets/catalog/mnist](https://www.tensorflow.org/datasets/catalog/mnist)**

## Laden der Daten
**Autor:** *NF*

Die Daten werden direkt über TensorFlow wie folgt abgerufen:

```python
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels
```
Die anschließende Datentransformation und -vorbereitung wird in einem späteren Abschnitt der Dokumentation eingehend betrachtet.

## Installation und Einrichtung
**Autor:** *NF*

### GPU-Unterstützung für TensorFlow

Damit TensorFlow auch dedizierte Grafikkarten (GPUs) für Berechnungen verwendet, müssen eventuell einige Konfigurationsschritte unternommen werden. Dies kann insbesondere unter Systemen mit Windows herausfordernd sein. Es wird empfohlen, bei der offiziellen [TensorFlow-Dokumentation](https://www.tensorflow.org/install/pip) zu beginnen, um TensorFlow für die GPU-Nutzung zu konfigurieren.

### Umgebung und Abhängigkeiten

Für die Entwicklung dieses Projekts wurde eine virtuelle Umgebung mit Anaconda erstellt. Es wird empfohlen, für die Reproduzierbarkeit der Ergebnisse und zur Vermeidung von Konflikten mit anderen Projekten eine ähnliche Umgebung zu verwenden.

#### Schritt 1: Anaconda Installation
Falls noch nicht installiert, laden Sie Anaconda von der [offiziellen Seite](https://www.anaconda.com/) herunter und folgen Sie den Installationsanweisungen für Ihr Betriebssystem.

#### Schritt 2: Erstellen einer virtuellen Umgebung

Erstellen Sie eine neue virtuelle Umgebung z.B. tensorflow-gpu (oder ein anderer präferierter Name) mit Python 3.8:

```bash
$ conda create -n tensorflow-gpu python=3.8
```

Aktivieren Sie die neu erstellte Umgebung:

```bash
$ conda activate tensorflow-gpu
```

#### Schritt 3: Installation der benötigten Pakete

Folgende Packages werden für unser Modell benötigt. Die abhängigen Packages sollten automatisch durch pip isntalliert werden.

| Name                   | Version              | Build            | Channel     |
|------------------------|----------------------|------------------|-------------|
| tensorflow             | 2.5.0                | pypi_0           | pypi        |
| matplotlib             | 3.6.0                | pypi_0           | pypi        |
| scikit-learn           | 1.3.2                | pypi_0           | pypi        |

Führen Sie für die Installation den folgenden Befehl in Ihrem (Anaconda-)Terminal bei aktiviertem "tensorflow-gpu" (o.Ä.) Enviroment aus:

```bash
$ pip install tensorflow==2.5 matplotlib==3.6 scikit-learn==1.3.2
```

### Prüfen ob TensorFlow die GPU nutzt

GPUs setzen sich immer mehr als Standard für Deep Learning durch. Dank ihrer großen Anzahl an logischen Kernen ermöglichen GPUs eine weitreichende Parallelverarbeitung, was sie im Vergleich zu herkömmlichen CPUs leistungsfähiger und schneller bei der Durchführung von Berechnungen macht.

Es ist gut möglich, dass in Ihrem Computer eine dedizierte Grafikkarte eingebaut ist. Allerdings kann es vorkommen, dass diese bei der Modellberechnung mit TensorFlow nicht zum Einsatz kommt. Standardmäßig erfolgt das Training auf der CPU, falls nicht anders konfiguriert. Deshalb ist es wichtig, zu überprüfen, ob TensorFlow tatsächlich Ihre GPU nutzt, um das volle Potenzial Ihres Systems auszuschöpfen.

Wenn Sie wissen wollen, ob TensorFlow die GPU-Beschleunigung nutzt oder nicht, können Sie einfach den folgenden Pythoncode verwenden:

````python
import tensorflow as tf

tf.config.list_physical_devices('GPU')
````
Die Ausgabe sollte dann in etwa so aussehen:

```console
2024-03-13 11:26:51.441234: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2024-03-13 11:27:06.093329: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library nvcuda.dll
2024-03-13 11:27:07.213670: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1060 with Max-Q Design computeCapability: 6.1
coreClock: 1.48GHz coreCount: 10 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 178.99GiB/s
2024-03-13 11:27:07.228018: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
2024-03-13 11:27:08.675544: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
2024-03-13 11:27:08.679613: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
2024-03-13 11:27:09.119743: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cufft64_10.dll
2024-03-13 11:27:09.201428: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library curand64_10.dll
2024-03-13 11:27:09.342143: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusolver64_11.dll
2024-03-13 11:27:09.493343: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusparse64_11.dll
2024-03-13 11:27:09.563140: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
2024-03-13 11:27:09.573457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
```

## Anleitung zur Benutzung
**Autor:** *NF*
### Daten laden und vorverarbeiten

Die Daten werden automatisch vom MNIST-Datensatz geladen und vorverarbeitet. Dies geschieht durch Ausführung der Skripte `data_loader.py` und `data_preprocess.py`, die für das Laden der Daten und deren Vorverarbeitung zuständig sind. Diese Schritte werden intern von der `main.py` Datei verwaltet, sodass keine manuelle Intervention erforderlich ist.

### Modelltraining

Das Training des Modells wird durch das Skript `model_training.py` verwaltet. Es definiert das CNN-Modell, führt das Training durch und wendet Methoden wie Early Stopping und Learning Rate Reduction an, um die Leistung des Modells zu optimieren.

Um das Training zu starten, führen Sie einfach die `main.py`-Datei aus:

```bash
$ python main.py
```


## Ergebnisse der CNN-Entwicklung für den MNIST-Datensatz
**Autor:** *NF*

Im Rahmen des Moduls Künstliche Intelligenz I stellten wir uns der Herausforderung ein Convolutional Neural Network (CNN) für die Handschriftenerkennung auf Basis des MNIST-Datensatzes zu entwickeln. Ziel war es, ein Modell zu entwerfen, das nicht nur präzise Vorhersagen trifft, sondern auch im Vergleich zu den in der Vorlesung behandelten Netzwerken überlegen ist.

Dabei haben wir für unser Modell bei unserem zweiten Gruppentreffen das gemeinsame Ziel wie folgt spezifiziert:

>Das CNN soll eine **Genauigkeit** von über 99,5 % auf den Testdatensatz erreichen, bei gleichzeitiger **Vermeidung von Overfitting** und ohne erheblichen **Rechenaufwand**. 

Nach dem Training wird das Modell automatisch mit dem Testdatensatz bewertet, und die Testverlust und -genauigkeit werden angezeigt. Darüber hinaus wird der Trainingsverlauf, einschließlich Trainings- und Validierungsverlust sowie -genauigkeit, mit Hilfe von Matplotlib geplottet. Diese Visualisierung hilft, die Leistung des Modells über die Zeit zu analysieren.

### Modellarchitektur

#### Modelle der Vorlesung
 Ausgangspunkt für die Architektur unseres CNN waren dabei die in der Vorlesung vorgestellten vier Modelle welche jeweils eine Genauigkeit auf den Testdatensatz zwischen 98,95 % und 99,27 % erreichten:
 
 | Name                   | Loss          | Accuracy         |
 |------------------------|---------------|------------------|
 | model_simple_conv      | 0,1163        | 0,9895           |
 | model_drop_conv        | 0,0359        | 0,9921           |
 | model_drop_move_conv   | 0,0293        | 0,9921           |
 | model_drop_full_conv   | 0,0250        | 0,9927           |

Das erste Modell in der Reihe ist ein einfaches CNN, welches aus einer *Convolutional Layer*, einer *MaxPooling Layer* zur Reduzierung der Dimensionalität, einer *Flatten Layer* zur Umwandlung der 2D-Ausgaben in einen Vektor, gefolgt von einer *Dense Layer* für die Klassifizierung und einer *Output Layer* mit *Softmax-Aktivierung*.

Das zweite Modell baut auf dem ersten auf, indem es eine *Dropout-Layer* einführt, um das Overfitting zu reduzieren. Durch das zufällige Deaktivieren von Neuronen während des Trainings wird eine robustere und generalisierbare Modellstruktur geschaffen. Diese zusätzliche Regularisierungsschicht soll die Modellperformance auf unbekannten Daten verbessern, indem sie eine bessere Generalisierung fördert.

Das dritte Modell erweitert das Konzept weiter durch Einführung von *Data Augmentation*. Zusätzlich zur Dropout-Layer wird das Modell mit einem Datensatz trainiert, der durch das Verschieben der ursprünglichen Bilder in verschiedene Richtungen vergrößert wurde. Diese Technik zielt darauf ab, die Robustheit des Modells gegenüber Verschiebungen und Veränderungen in den Eingabedaten zu erhöhen, indem es lernt, wichtige Merkmale unabhängig von ihrer Position im Bild zu erkennen.

Schließlich integriert das vierte Modell neben der Dropout-Regulierung und den Verschiebungen auch Rotationen in die Data Augmentation, um das Trainingsspektrum noch weiter zu erweitern. Dabei werden die Bilder zufällig zwischen -30° und +30° gedreht. Diese Kombination von Techniken zielt darauf ab, das CNN noch flexibler und widerstandsfähiger gegenüber einer Vielzahl von Bildvariationen zu machen. Durch das Training mit einem so diversifizierten Datensatz strebt das vierte Modell die höchste Generalisierungsfähigkeit und Leistung unter den vier Ansätzen an.

#### Unser Modell
Unser Modell besteht aus mehreren *Convolutional* (*Conv2D*) und *MaxPooling2D*-Schichten, gefolgt von *Dropout*-Schichten, einer *Flatten*-Schicht und mehreren *Dense*-Schichten, die in eine *Softmax*-Aktivierungsfunktion münden.

Die Architektur beginnt mit *Conv2D*-Schichten, die die Grundlage der Feature-Extraktion bilden. Diese Schichten wenden Filter auf die Eingabebilder an, um Merkmale wie Kanten, Ecken und Texturen zu identifizieren. Die Nutzung von 5x5-Filtern gefolgt von 3x3-Filtern ermöglicht es dem Netzwerk, zunächst breitere Muster zu erkennen und daraufhin detailliertere, feinere Strukturen zu erfassen.

Um den Lernprozess weiter zu optimieren und zu stabilisieren, folgt auf jede Conv2D-Schicht eine BatchNormalization. Wir haben durch Experimentieren festgestellt, dass durch die BatchNormalization nach jedem Schritt, dass Modell schneller während des Trainings konvergiert und weniger empfindlich auf kleine Schwankungen in den Trainingsdaten reagiert. Das Resultat ist eine Reduzierung des Overfittings.

Zur Reduktion der Komplexität und weiteren Vermeidung von Overfitting tragen MaxPooling2D-Schichten bei. Sie verringern die räumlichen Dimensionen der Feature-Maps, wodurch die Anzahl der Parameter reduziert und die Rechenanforderungen gesenkt werden. Diese Verringerung zwingt das Netzwerk, sich auf die wesentlichsten Merkmale zu konzentrieren, was die Generalisierungsfähigkeit des Modells verbessert.

Ein weiterer wichtiger Aspekt der Architektur ist die Integration von Dropout als Regularisierungstechnik. Durch das zufällige Nullsetzen von Neuronenausgaben während des Trainings wird eine zu starke Abhängigkeit von den Trainingsdaten vermieden. Diese Methode fördert eine robuste Feature-Erkennung, die auf neuen, unbekannten Daten gut generalisiert.
Schließlich wird die Architektur durch Flatten- und Dense-Schichten vervollständigt, die die extrahierten und verarbeiteten Merkmale in Klassifikationswahrscheinlichkeiten umwandeln. Die Flatten-Schicht konvertiert die 2D-Feature-Maps in einen eindimensionalen Vektor, der von den Dense-Schichten verarbeitet wird. Die letzte Schicht nutzt die Softmax-Funktion, um die Zugehörigkeit eines Bildes zu einer der 10 Ziffernklassen in Form von Wahrscheinlichkeiten auszudrücken. 

## Konfigurationsmöglichkeiten
**Autor:** *NF*

Das Projekt bietet verschiedene Konfigurationsmöglichkeiten über die `main.py`-Datei:

- `TEST_SIZE`: Legt den Anteil der Validierungsdaten fest.
- `epochs`: Die maximale Anzahl der Trainingsepochen. Das Training kann aufgrund des Early Stoppings früher beendet werden.
- `batch_size`: Die Größe der Batches während des Trainings.
- `SEED`: Der Seed für den Zufallsgenerator, der für die Datenaufteilung verwendet wird.
- `early_stopping_enabled`: Aktiviert oder deaktiviert das Early Stopping.

Diese Einstellungen können direkt in der `main.py`-Datei angepasst werden, um das Training nach Bedarf zu konfigurieren.

## Dateistruktur
**Autor:** *NF*

Das Projekt "CNN-HANDWRITING" besteht aus den folgenden Dateien:

- `README.md`: Diese Datei. Sie enthält eine Einführung in das Projekt, Installationsanweisungen, Nutzungsinformationen und eine Beschreibung der Dateistruktur.
- `best_parameters.json`: Eine JSON-Datei, die automatisch erstellt wird, um die besten Parameter und Metriken des Trainings zu speichern. Diese Datei wird von `data_loader.py` verwendet, um bei zukünftigen Trainingsdurchläufen die besten Parameter zu laden und anzuwenden.
- `data_loader.py`: Enthält Funktionen zum Laden des MNIST-Datensatzes und zum Speichern/Laden der besten Trainingsparameter.
- `data_preprocess.py`: Beinhaltet die Vorverarbeitungslogik für die Bilddaten, einschließlich der Aufteilung in Trainings- und Validierungsdatensätze und der Normalisierung.
- `main.py`: Der Hauptskript, der den gesamten Trainingsprozess orchestriert, von Datenladung und -vorverarbeitung bis hin zum Modelltraining und der Ergebnisanalyse.
- `model_training.py`: Definiert das CNN-Modell und die Trainingsroutine, einschließlich Callbacks wie Early Stopping und Learning Rate Reduction.
- `my_plot.py`: Bietet Funktionen zum Plotten von Trainings- und Validierungsverlust sowie Genauigkeit über die Epochen.

Um das Projekt auszuführen und das Modell zu trainieren, starten Sie einfach `main.py` mit Python. Stellen Sie sicher, dass alle Abhängigkeiten gemäß den Anweisungen in "Installation und Abhängigkeiten" installiert sind.

## Ausblick/To-Do
**Autor:** *NF*
 - Ensemble Learning