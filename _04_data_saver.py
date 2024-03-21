# _04_data_saver.py
from tensorflow.keras.datasets import mnist
import json

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
        return True
    else:
        print("Die aktuellen Metriken sind nicht besser als die bisher Besten.")
        return False
    

def update_readme_from_json(json_file_path, readme_file_path, start_marker, end_marker):
    """
    Aktualisiert einen spezifischen Abschnitt der README.md-Datei basierend auf den Inhalten einer JSON-Datei.
    
    Parameters:
    json_file_path (str): Der Pfad zur JSON-Datei.
    readme_file_path (str): Der Pfad zur README.md-Datei.
    start_marker (str): Marker, der den Beginn des zu aktualisierenden Abschnitts kennzeichnet.
    end_marker (str): Marker, der das Ende des zu aktualisierenden Abschnitts kennzeichnet.
    """

    # JSON-Datei laden und lesen
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Inhalte für den zu aktualisierenden Abschnitt formatieren
    formatted_content = ""
    for key, value in data.items():
        formatted_content += f"**{key}:** {value}\n\n"

    # README.md-Datei lesen
    with open(readme_file_path, 'r') as readme_file:
        readme_contents = readme_file.readlines()

    # Finden der Marker und aktualisieren des Inhalts
    start_index = end_index = None
    for i, line in enumerate(readme_contents):
        if start_marker in line:
            start_index = i + 1
        elif end_marker in line and start_index is not None:
            end_index = i
            break

    if start_index is not None and end_index is not None:
        updated_readme_contents = readme_contents[:start_index] + [formatted_content] + readme_contents[end_index:]
        # README.md-Datei mit aktualisierten Inhalten schreiben
        with open(readme_file_path, 'w') as readme_file:
            readme_file.writelines(updated_readme_contents)
        print("README.md wurde erfolgreich aktualisiert.")
    else:
        print("Marker nicht gefunden.")