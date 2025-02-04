import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np


def plot_ecg_signal(signal, fs, segment_duration=30):
    """
    Rysuje i zapisuje wykresy EKG w segmentach o określonym czasie.

    :param signal: Tablica wartości sygnału EKG.
    :param fs: Częstotliwość próbkowania sygnału.
    :param segment_duration: Długość każdego segmentu w sekundach.
    """
    time = np.arange(len(signal)) / fs  # Oś czasu w sekundach
    total_duration = len(signal) / fs  # Całkowity czas w sekundach
    num_segments = int(np.ceil(total_duration / segment_duration))
    save_folder = "visualization_temp_signal"
    os.makedirs(save_folder, exist_ok=True)

    for j in range(num_segments):
        start_idx = int(j * segment_duration * fs)
        end_idx = int(min((j + 1) * segment_duration * fs, len(signal)))

        plt.figure(figsize=(100, 4))
        plt.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color="b", linewidth=1)
        plt.xlabel("Czas (s)")
        plt.ylabel("Napięcie (mV)")
        plt.title(f"Segment {j+1}")

        save_path = os.path.join(save_folder, f"ecg_segment_{j+1}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📁 Wykres zapisany: {save_path}")
        plt.close()



def plot_ecg_record(record_path, selected_channel=None, segment_duration=30):
    """
    Rysuje i zapisuje wykresy EKG w segmentach o określonym czasie.

    :param record_path: Ścieżka do rekordu, np. 'data/raw/mitdb/100'
    :param selected_channel: Nazwa kanału, np. 'MLII' lub 'V5'. Jeśli None, wybiera pierwszy dostępny.
    :param segment_duration: Długość każdego segmentu w sekundach.
    :param image_width: Szerokość każdego obrazka w pikselach.
    """
    # Wczytanie rekordu i adnotacji
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    # Pobranie danych sygnałowych
    signals = record.p_signal
    leads = record.sig_name  # Nazwy odprowadzeń
    fs = record.fs  # Częstotliwość próbkowania

    # Wybór kanału
    if selected_channel and selected_channel in leads:
        channel_idx = leads.index(selected_channel)
    else:
        channel_idx = 0  # Domyślnie wybiera pierwszy kanał

    signal = signals[:, channel_idx]  # Pobranie wybranego kanału
    time = np.arange(signal.shape[0]) / fs  # Oś czasu w sekundach

    # Pobranie adnotacji
    annotation_positions = annotation.sample / fs  # Konwersja na sekundy
    annotation_labels = annotation.symbol

    # Ustalenie liczby segmentów
    total_duration = len(signal) / fs  # Całkowity czas w sekundach
    num_segments = int(np.ceil(total_duration / segment_duration))

    # Tworzenie folderu na wykresy
    save_folder = "visualization_temp_record"
    os.makedirs(save_folder, exist_ok=True)

    for j in range(num_segments):
        start_idx = int(j * segment_duration * fs)
        end_idx = int(min((j + 1) * segment_duration * fs, len(signal)))

        plt.figure(figsize=( 100 , 4 ))  # Konwersja szerokości na cal (100 dpi)
        plt.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color="b", linewidth=1)

        # Dodawanie adnotacji
        for pos, label in zip(annotation_positions, annotation_labels):
            if time[start_idx] <= pos <= time[end_idx - 1]:
                plt.scatter(pos, signal[int(pos * fs)], color='red', marker='o')
                plt.text(pos, signal[int(pos * fs)], label, fontsize=10, verticalalignment='bottom', color='red')

        # Zapisywanie wykresu
        save_path = os.path.join(save_folder, f"{os.path.basename(record_path)}_segment_{j+1}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📁 Wykres zapisany: {save_path}")

        plt.close()  # Zamknięcie wykresu po zapisie



def plot_full_ecg_record(record_name, signal, annotations, labels, fs=360, start_time=None, end_time=None, padding=2):
    """
    Rysuje wykres EKG z buforem i poprawionym układem wykresu.

    :param record_name: Nazwa rekordu
    :param signal: Sygnał EKG (pojedynczy kanał)
    :param annotations: Lista pozycji próbek z adnotacjami
    :param labels: Lista etykiet anomalii
    :param fs: Częstotliwość próbkowania (np. 360 Hz)
    :param start_time: Początek wycinka (w sekundach) - domyślnie od początku
    :param end_time: Koniec wycinka (w sekundach) - domyślnie do końca
    :param padding: Dodatkowy bufor po lewej i prawej stronie (w sekundach)
    """

    # 🔹 Jeśli `start_time` lub `end_time` nie są podane, ustawiamy pełny zakres
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = len(signal) / fs  # Cały sygnał

    # 🔹 Konwersja czasu na próbki
    start_sample = max(0, int((start_time - padding) * fs))  # Uwzględniamy padding (ale nie poniżej 0)
    end_sample = min(len(signal), int((end_time + padding) * fs))  # Uwzględniamy padding

    # 🔹 Wycinamy fragment sygnału
    time_axis = np.arange(start_sample, end_sample) / fs
    signal_cut = signal[start_sample:end_sample]

    # 🔹 Tworzenie wykresu z poprawionym paddingiem
    fig, ax = plt.subplots(figsize=(310, 4))
    ax.plot(time_axis, signal_cut, color='black', linewidth=1, label="Sygnał EKG")

    # 🔹 Poprawa marginesów wokół wykresu
    ax.margins(0.05)  # Dodaje przestrzeń wokół wykresu
    plt.subplots_adjust(left=0.07, right=0.80, top=0.9, bottom=0.2)  # Padding

    # 🔹 Mapowanie kolorów dla adnotacji
    label_colors = {
        'N': 'green', 'V': 'purple', 'Q': 'brown', 'A': 'cyan', 'F': 'orange',
        'f': 'blue', 'S': 'lime', 'J': 'gold', 'L': 'pink', 'R': 'darkblue',
        'j': 'lightblue', 'E': 'red', '/': 'magenta', '|': 'yellow',
        '~': 'gray', '!': 'crimson', 'a': 'olive', 'x': 'darkred',
        '[': 'black', ']': 'black', '"': 'teal', 'e': 'darkgreen'
    }

    label_markers = {
        'N': 'o',  # ✅ Normalne pobudzenie QRS (standardowy skurcz serca)
        'V': 'D',  # ⚠️ Przedwczesne pobudzenie komorowe (PVC) – nieprawidłowe skurcze komorowe
        'Q': 'x',  # ❓ Nieokreślona arytmia (może być różnymi rodzajami zaburzeń)
        'A': '^',  # 🔼 Pobudzenie nadkomorowe (SVEB) – nadkomorowe pobudzenie ektopowe
        'F': '*',  # ⭐ Fusion beat (pobudzenie mieszane) – skurcz będący wynikiem dwóch sygnałów
        'f': 'v',  # 🔽 Flutter przedsionkowy – szybkie, ale regularne drgania przedsionków
        'S': 's',  # 🟢 Nadkomorowe pobudzenie ektopowe – inny rodzaj SVEB
        'J': 'h',  # 🔄 Junctional escape beat – pobudzenie z okolicy węzła przedsionkowo-komorowego
        'L': 'p',  # ⬅️ Lewogram – przemieszczenie osi serca w lewo
        'R': 'H',  # ➡️ Prawogram – przemieszczenie osi serca w prawo
        'j': '8',  # 🟠 Junkcyjne pobudzenie – z obszaru węzła AV
        'E': 'P',  # ❌ Ektopowe pobudzenie komorowe – bardzo nieprawidłowe pobudzenie komorowe
        '/': 'X',  # 🔁 Separator nowego segmentu rytmu
        '|': '|',  # 🔀 Separator rytmu – wskazuje początek nowego rytmu serca
        '~': '_',  # 🔊 Artefakt – zakłócenia w sygnale (np. ruch elektrody, zakłócenia elektryczne)
        '!': '+',  # 🚀 Tachykardia – szybkie bicie serca (>100 BPM)
        'a': '<',  # ⏪ Aberrant beat – nietypowe pobudzenie nadkomorowe
        'x': '>',  # ❓ Nieznana anomalia – rzadko spotykane pobudzenie (brak oficjalnej dokumentacji)
        '[': '1',  # 📌 Otworzenie segmentu – początek analizy odcinka sygnału
        ']': '2',  # 📌 Zamknięcie segmentu – koniec analizy odcinka sygnału
        '"': '3',  # 📝 Specyficzne zdarzenie analizy (np. zmiana ustawień detekcji)
        'e': '4'   # 🛑 Escape beat – pobudzenie ratunkowe serca (gdy naturalny rytm zwalnia)
    }


    # 🔹 Rysowanie adnotacji w wybranym zakresie czasu
    used_labels = set()
    for ann, label in zip(annotations, labels):
        ann_time = ann / fs
        if start_time - padding <= ann_time < end_time + padding and label in label_colors:
            color = label_colors[label]
            marker = label_markers[label]

            # Rysowanie oznaczenia
            ax.scatter(ann_time, signal[ann], color=color, marker=marker, s=80, label=label if label not in used_labels else "")
            used_labels.add(label)

    # 🔹 Konfiguracja wykresu
    ax.set_title(f"Sygnał EKG: {record_name} ({start_time}s - {end_time}s)")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")

    # **Nowa szerokość z paddingiem**
    ax.set_xlim(start_time - padding, end_time + padding)

    # 🔹 Poprawa czytelności: legenda poza wykresem
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)  # Przesunięcie legendy poza wykres

    # 🔹 Eksport do pliku SVG
    plt.savefig(f"{record_name}_{start_time}s_{end_time}s_ekg.svg", format='svg', bbox_inches='tight')

    plt.show()



def visualize_interpolation(segment_resized, new_annotations, num_samples):
    """
    Wizualizuje interpolowany fragment EKG wraz z nowymi adnotacjami.

    :param segment_resized: Interpolowany fragment sygnału EKG
    :param new_annotations: Nowe pozycje adnotacji po interpolacji
    :param num_samples: Docelowa liczba próbek w interpolacji
    :param segment_index: Indeks segmentu w całym sygnale
    """
    x_interpolated = np.linspace(0, 1, num_samples)

    plt.figure(figsize=(10, 5))
    plt.plot(x_interpolated, segment_resized, label="Interpolowany", color="orange", linewidth=1.5)


    # **Dodanie adnotacji jako zielone kropki**
    annotation_x = new_annotations / num_samples  # Normalizacja pozycji adnotacji
    annotation_y = segment_resized[new_annotations]  # Pobranie wartości amplitudy

    plt.scatter(annotation_x, annotation_y, color='green', marker='o', label="Nowe adnotacje")

    plt.legend()
    plt.title(f"Interpolacja segmentu QRS")
    plt.xlabel("Normalizowany czas")
    plt.ylabel("Amplituda")
    plt.grid()
    plt.show()


