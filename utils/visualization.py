import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from config import NUM_SAMPLES, FS


def plot_full_ecg_record(record_name, signal, annotations, labels, fs=FS, start_time=None, end_time=None, padding=1):
    """
    Rysuje i zapisuje wykres EKG z adnotacjami do pliku.

    :param record_name: Nazwa rekordu (będzie częścią nazwy pliku).
    :param signal: Sygnał EKG (pojedynczy kanał).
    :param annotations: Lista pozycji próbek z adnotacjami.
    :param labels: Lista etykiet anomalii.
    :param fs: Częstotliwość próbkowania (np. 360 Hz).
    :param start_time: Początek zakresu (sekundy). Jeśli `None`, bierze całość.
    :param end_time: Koniec zakresu (sekundy). Jeśli `None`, bierze całość.
    :param padding: Dodatkowy bufor po lewej i prawej stronie (w sekundach).
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

    # 🔹 Tworzenie folderu na wykresy
    save_folder = "utils/whole_subplots_visualization_folder"
    os.makedirs(save_folder, exist_ok=True)

    # 🔹 Tworzenie wykresu
    plt.figure(figsize=(300, 4))
    plt.plot(time_axis, signal_cut, color='black', linewidth=1, label="Sygnał EKG")

    # 🔹 Dodawanie adnotacji
    annotation_positions = np.array(annotations) / fs
    for ann, label in zip(annotation_positions, labels):
        if start_time - padding <= ann < end_time + padding:
            plt.scatter(ann, signal[int(ann * fs)], color='red', marker='o')
            plt.text(ann, signal[int(ann * fs)], label, fontsize=10, verticalalignment='bottom', color='red')

    # 🔹 Konfiguracja osi i zapis do pliku
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title(f"Sygnał EKG: {record_name} ({start_time}s - {end_time}s)")
    plt.legend()

    save_path = os.path.join(save_folder, f"{record_name}_{start_time}s_{end_time}s.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()  # 🔥 Zamknięcie wykresu, oszczędza RAM

    print(f"📁 Wykres zapisany: {save_path}")




def visualize_interpolation(signal, annotations=None, segment_id=0, patient_name="unknown"):
    """
    Rysuje i zapisuje interpolowany segment EKG z naniesionymi adnotacjami.

    :param signal: Interpolowany sygnał EKG (1D numpy array).
    :param annotations: Lista indeksów adnotacji po interpolacji.
    :param segment_id: Numer segmentu, do nazwy pliku.
    :param patient_name: Nazwa pacjenta do personalizacji plików.
    """

    save_folder = "utils/visualization_folder"
    os.makedirs(save_folder, exist_ok=True)  # 🔥 Tworzy folder jeśli nie istnieje

    plt.figure(figsize=(10, 4))
    plt.plot(signal, color="b", linewidth=1, label="Interpolowany sygnał")

    # 🔹 Adnotacje
    if annotations is not None and len(annotations) > 0:
        for ann in annotations:
            plt.scatter(ann, signal[ann], color='red', marker='o')

    plt.xlabel("Próbki")
    plt.ylabel("Znormalizowana wartość")
    plt.title(f"Interpolowany segment EKG ({patient_name} - {segment_id})")
    plt.legend()

    # 🔥 Zapis do pliku z nazwą pacjenta i segmentu
    save_path = os.path.join(save_folder, f"{patient_name}_segment_{segment_id}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()  # 🔥 Zamyka wykres, oszczędza RAM

    print(f"📁 Wykres zapisany: {save_path}")



def visualize_plot_interpolation(signal, annotations=None, segment_id=0, patient_name="unknown"):
    """
    Rysuje i zapisuje interpolowany segment EKG z naniesionymi adnotacjami.

    :param signal: Interpolowany sygnał EKG (1D numpy array).
    :param annotations: Lista indeksów adnotacji po interpolacji.
    :param segment_id: Numer segmentu, do nazwy pliku.
    :param patient_name: Nazwa pacjenta do personalizacji plików.
    """

    plt.figure(figsize=(10, 4))
    plt.plot(signal, color="b", linewidth=1, label="Interpolowany sygnał")

    # 🔹 Adnotacje
    if annotations is not None and len(annotations) > 0:
        for ann in annotations:
            plt.scatter(ann, signal[ann], color='red', marker='o')

    plt.xlabel("Próbki")
    plt.ylabel("Znormalizowana wartość")
    plt.title(f"Interpolowany segment EKG ({patient_name} - {segment_id})")
    plt.legend()

    # ✅ Wyświetlenie wykresu na ekranie
    plt.show()



