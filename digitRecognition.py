import numpy as np
import PySimpleGUI as sg
import os
import json
import random
import matplotlib.pyplot as plt  # Biblioteka do wykresów

# Klasa perceptronu
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Wagi perceptronu zainicjalizowane losowo
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def activation(self, x):
        # Funkcja aktywacji (sigmoidalna)
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        # Przewidywanie na podstawie sumy ważonej i funkcji aktywacji
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if self.activation(weighted_sum) >= 0.5 else 0

    def train(self, training_inputs, labels, epochs=40):
        # Lista do przechowywania błędów dla każdej epoki
        errors_per_epoch = []

        for epoch in range(epochs):
            epoch_errors = 0  # Licznik błędów w bieżącej epoce
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                epoch_errors += abs(error)  

            errors_per_epoch.append(epoch_errors)  # Zapisujemy błąd epoki

        return errors_per_epoch  # Zwracamy listę błędów

def get_digit_patterns():
    # Cyfry w postaci macierzy 5x7
    zero = np.array([1, 1, 1, 1, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1,
                     1, 1, 1, 1, 1])

    one = np.array([0, 0, 1, 0, 0, 
                    0, 1, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 1, 1, 1, 0])

    two = np.array([1, 1, 1, 1, 1, 
                    0, 0, 0, 0, 1, 
                    0, 0, 0, 0, 1, 
                    1, 1, 1, 1, 1, 
                    1, 0, 0, 0, 0, 
                    1, 0, 0, 0, 0, 
                    1, 1, 1, 1, 1])

    three = np.array([1, 1, 1, 1, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      1, 1, 1, 1, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      1, 1, 1, 1, 1])

    four = np.array([1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 1, 
                     0, 0, 0, 0, 1, 
                     0, 0, 0, 0, 1])

    five = np.array([1, 1, 1, 1, 1, 
                     1, 0, 0, 0, 0, 
                     1, 0, 0, 0, 0, 
                     1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 1, 
                     0, 0, 0, 0, 1, 
                     1, 1, 1, 1, 1])

    six = np.array([1, 1, 1, 1, 1, 
                    1, 0, 0, 0, 0, 
                    1, 0, 0, 0, 0, 
                    1, 1, 1, 1, 1, 
                    1, 0, 0, 0, 1, 
                    1, 0, 0, 0, 1, 
                    1, 1, 1, 1, 1])

    seven = np.array([1, 1, 1, 1, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1, 
                      0, 0, 0, 0, 1])

    eight = np.array([1, 1, 1, 1, 1, 
                      1, 0, 0, 0, 1, 
                      1, 0, 0, 0, 1, 
                      1, 1, 1, 1, 1, 
                      1, 0, 0, 0, 1, 
                      1, 0, 0, 0, 1, 
                      1, 1, 1, 1, 1])

    nine = np.array([1, 1, 1, 1, 1, 
                     1, 0, 0, 0, 1, 
                     1, 0, 0, 0, 1, 
                     1, 1, 1, 1, 1, 
                     0, 0, 0, 0, 1, 
                     0, 0, 0, 0, 1, 
                     1, 1, 1, 1, 1])

    return [zero, one, two, three, four, five, six, seven, eight, nine]
# Zapis cyfr do pliku
def save_to_file(digit, label, filename='digits_data.json'):
    digit = digit.tolist()  # Przekształcamy na listę
    data = {"digits": []}
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

    data["digits"].append({"digit": digit, "label": label})

    with open(filename, 'w') as f:
        json.dump(data, f)

# Wczytanie zapisanych cyfr
def load_data(filename='digits_data.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            for entry in data["digits"]:
                entry["digit"] = np.array(entry["digit"])
            return data["digits"]
    return []

# Inicjalizacja wzorców cyfr i perceptronów
digits = get_digit_patterns()
perceptrons = [Perceptron(input_size=35, learning_rate=0.1) for _ in range(10)]

# Trenowanie perceptronów
errors_per_perceptron = []
for i in range(10):
    train_labels = np.zeros(10)  # Wektor etykiet dla każdej cyfry
    train_labels[i] = 1  # Tylko dla jednej cyfry wartość będzie 1
    errors = perceptrons[i].train(digits, train_labels, epochs=100)  # Zwiększamy liczbę epok
    errors_per_perceptron.append(errors)

# Generowanie wykresów błędów dla każdego perceptronu
plt.figure(figsize=(12, 6))
for i, errors in enumerate(errors_per_perceptron):
    plt.plot(errors, label=f"Perceptron {i}")

plt.title("Błędy uczenia w zależności od epoki")
plt.xlabel("Epoka")
plt.ylabel("Liczba błędów")
plt.legend()
plt.show()

# Inicjalizacja GUI
sg.theme("DarkAmber")
layout = [
    [sg.Button('', size=(2, 1), key=f'CELL_{row}_{col}', pad=(1, 1), button_color=('white', 'black')) for col in range(5)] for row in range(7)
]
layout.append([sg.Button('Rozpoznaj', size=(10, 1))])
layout.append([sg.Button('Rozpoznaj z szumem', size=(15, 1))])
layout.append([sg.Button('Usuń wszystkie cyfry', size=(20, 1))]) 
layout.append([sg.Text('', size=(30, 1), key='RESULT')])
layout.append([sg.Button('Dodaj cyfrę do zbioru uczącego', size=(20, 1))])
layout.append([sg.Button('Pokaż zapisane cyfry', size=(20, 1))])

window = sg.Window('Rozpoznawanie cyfr', layout)

# Inicjalizacja siatki 7x5
grid = np.zeros((7, 5))

# Funkcja do aktualizacji przycisków w siatce
def update_button(row, col):
    current_value = grid[row, col]
    new_value = 1 - current_value
    grid[row, col] = new_value
    color = 'white' if new_value == 0 else 'green'
    window[f'CELL_{row}_{col}'].update(button_color=('white', color))

# Funkcja wyświetlająca wyniki perceptronów
def print_perceptron_outputs(input_vector):
    print("Wyniki perceptronów dla bieżącego wzorca cyfry:")
    for i in range(10):
        prediction = perceptrons[i].predict(input_vector)
        print(f"Perceptron {i}: {prediction}")

# Funkcja wyboru etykiety cyfry
def choose_digit_label():
    layout = [
        [sg.Text('Wybierz cyfrę, którą chcesz dodać do zbioru uczącego')],
        [sg.Button(str(i)) for i in range(10)],
        [sg.Button('Anuluj')]
    ]
    window = sg.Window('Wybór cyfry', layout)
    event, _ = window.read()
    window.close()

    if event == sg.WIN_CLOSED or event == 'Anuluj':
        return None
    else:
        return int(event)

def delete_all_digits():

    response = sg.popup_yes_no("Czy na pewno chcesz usunąć wszystkie zapisane cyfry?")
    
    if response == 'Yes':
        save_all_data([])  # Czyszczenie pliku ze wszystkimi zapisanymi cyframi
        sg.popup("Wszystkie zapisane cyfry zostały usunięte.")
    else:
        sg.popup("Operacja anulowana.")
# Funkcja do usuwania wszystkich zapisanych cyfr z pliku
def save_all_data(data, filename='digits_data.json'):
    
    with open(filename, 'w') as f:
        json.dump({"digits": data}, f)

# Funkcja dodająca nową cyfrę do zbioru uczącego
def add_new_digit_to_training_data(digit, digit_label, noise_level=0.03):
    # Zapisujemy czystą cyfrę
    save_to_file(digit.flatten(), digit_label)

    # Tworzymy zaszumioną wersję i zapisujemy ją
    noisy_digit = add_noise_to_digit(digit.flatten(), noise_level)
    save_to_file(noisy_digit, digit_label)
    
    return noisy_digit, digit_label

# Funkcja dodająca szum do cyfry
def add_noise_to_digit(digit, noise_level=0.05):
    noisy_digit = digit.copy()
    total_pixels = len(digit)
    num_noisy_pixels = int(noise_level * total_pixels)
    noisy_indices = random.sample(range(total_pixels), num_noisy_pixels)

    # Zmieniamy wartości losowych pikseli
    for idx in noisy_indices:
        noisy_digit[idx] = 1 - noisy_digit[idx]

    return noisy_digit
def add_noise_to_input(input_vector, noise_level=0.15):
    noisy_input = input_vector.copy()
    total_pixels = len(noisy_input)
    num_noisy_pixels = int(noise_level * total_pixels)
    noisy_indices = random.sample(range(total_pixels), num_noisy_pixels)

    # Zmieniamy wartości losowych pikseli
    for idx in noisy_indices:
        noisy_input[idx] = 1 - noisy_input[idx]

    return noisy_input    
def print_perceptron_outputs(input_vector):
    print("Wyniki perceptronów dla bieżącego wzorca cyfry:")
    predicted_digits = []
    for i in range(10):
        prediction = perceptrons[i].predict(input_vector)
        if prediction == 1:  # Jeśli perceptron wykrył cyfrę
            predicted_digits.append(i)
    if not predicted_digits:
        print("Brak rozpoznania cyfry.")
    else:
        print(f"Znalezione cyfry: {', '.join(map(str, predicted_digits))}")
# Funkcja wyświetlająca zapisane cyfry
def display_saved_digits():
    saved_digits = load_data()

    if not saved_digits:
        sg.popup("Brak zapisanych cyfr.")
        return

    layout_choose_digit = [
        [sg.Text("Wybierz cyfrę do wyświetlenia:")],
        [sg.Button(str(i), size=(3, 1)) for i in range(10)],
        [sg.Button("Wszystkie", size=(10, 1)), sg.Button("Anuluj", size=(10, 1))],
    ]
    window_choose_digit = sg.Window("Wybierz cyfrę", layout_choose_digit)
    event, _ = window_choose_digit.read()
    window_choose_digit.close()

    if event == "Anuluj" or event == sg.WIN_CLOSED:
        return

    selected_digit = int(event) if event != "Wszystkie" else None
    layout_display = []

    # Wyświetlamy wybraną cyfrę lub wszystkie zapisane
    for entry in saved_digits:
        if selected_digit is None or entry["label"] == selected_digit:
            digit = np.array(entry["digit"]).reshape((7, 5))
            digit_text = "\n".join(" ".join(str(int(cell)) for cell in row) for row in digit)
            label = entry["label"]
            layout_display.append([sg.Text(f"Cyfra: {label}", size=(5, 1)), sg.Text(digit_text, size=(20, 10))])

    if not layout_display:
        sg.popup(f"Brak zapisanych cyfr {selected_digit}.")
        return

    window_display = sg.Window(f"Zapisane cyfry {selected_digit if selected_digit is not None else 'wszystkie'}", layout_display)
    window_display.read()
    window_display.close()

# Pętla główna aplikacji
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event.startswith('CELL_'):
        _, row, col = event.split('_')
        row, col = int(row), int(col)
        update_button(row, col)

    elif event == 'Rozpoznaj':
        input_vector = grid.flatten()
        print_perceptron_outputs(input_vector)
        predictions = [p.predict(input_vector) for p in perceptrons]
        recognized_digits = [i for i, pred in enumerate(predictions) if pred == 1]  # Lista rozpoznanych cyfr
        if recognized_digits:
            window['RESULT'].update(f"Rozpoznano cyfry: {', '.join(map(str, recognized_digits))}")
        else:
            window['RESULT'].update("Nie rozpoznano żadnej cyfry.")

    elif event == 'Rozpoznaj z szumem':
        input_vector = grid.flatten()
        noisy_input = add_noise_to_input(input_vector, noise_level=0.05)  # Nałożenie szumu na obraz
        print_perceptron_outputs(noisy_input)  # Wyświetlenie wyników dla zaszumionej wersji
        predictions = [p.predict(noisy_input) for p in perceptrons]  # Predykcje na zaszumionej wersji
        recognized_digits = [i for i, pred in enumerate(predictions) if pred == 1]  # Lista rozpoznanych cyfr
        if recognized_digits:
            window['RESULT'].update(f"Rozpoznano cyfry z szumem: {', '.join(map(str, recognized_digits))}")
        else:
            window['RESULT'].update("Nie rozpoznano żadnej cyfry.")

    elif event == 'Dodaj cyfrę do zbioru uczącego':
        digit_label = choose_digit_label()
        if digit_label is not None:
            clean_digit = grid.flatten()
            noisy_digit, label = add_new_digit_to_training_data(grid, digit_label)

            for i in range(10):
                train_labels = np.zeros(10)
                if i == digit_label:
                    train_labels[i] = 1
                perceptrons[i].train(np.array([clean_digit, noisy_digit]), train_labels, epochs=10)

            sg.popup("Cyfra (czysta i zaszumiona) dodana do zbioru uczącego.")
        else:
            sg.popup("Operacja anulowana.")

    elif event == 'Pokaż zapisane cyfry':
        display_saved_digits()


    elif event == 'Usuń wszystkie cyfry':
        delete_all_digits()
