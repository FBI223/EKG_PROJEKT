import os
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import numpy as np
from config import NUM_SAMPLES


def normalize_signal(ecg_signal):
    """Normalizuje sygnał EKG"""
    return (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))

def filter_signal(ecg_signal):
    """Filtruje sygnał EKG usuwając szumy"""
    return medfilt(ecg_signal, kernel_size=5)


def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples=NUM_SAMPLES):
    """
    Interpoluje segment EKG do stałej liczby próbek, jednocześnie zachowując względne położenie adnotacji.

    :param segment: Oryginalny fragment sygnału EKG (numpy array)
    :param annotations: Lista adnotacji w oryginalnym segmencie
    :param segment_start: Indeks początku segmentu w oryginalnym sygnale
    :param segment_end: Indeks końca segmentu w oryginalnym sygnale
    :param num_samples: Docelowa liczba próbek w segmencie
    :return: Interpolowany segment, nowe pozycje adnotacji
    """
    if len(segment) == num_samples:
        return segment, np.array(annotations)  # Nic nie zmieniamy

    # Sprawdzenie, czy annotations to lista, jeśli nie - konwersja
    if isinstance(annotations, np.int64) or isinstance(annotations, int):
        annotations = [annotations]  # Konwersja do listy

    # Stworzenie siatki interpolacyjnej
    x_old = np.linspace(segment_start, segment_end, len(segment))
    x_new = np.linspace(segment_start, segment_end, num_samples)
    interpolator = interp1d(x_old, segment, kind='linear', fill_value="extrapolate")

    segment_resized = interpolator(x_new)

    # Przeskalowanie adnotacji do nowego przedziału
    new_annotations = []
    scale_factor = num_samples / (segment_end - segment_start)

    for ann in annotations:
        rel_pos = (ann - segment_start) * scale_factor
        new_annotations.append(int(rel_pos))  # Nowa pozycja adnotacji po interpolacji

    return segment_resized, np.array(new_annotations)




def extract_unique_annotations(directory="data/raw/mitdb/"):
    """
    Przechodzi przez wszystkie rekordy EKG w bazie i wyodrębnia unikalne adnotacje.

    :param directory: Ścieżka do katalogu z plikami MIT-BIH lub innej bazy
    :return: Słownik {adnotacja: liczba wystąpień}, lista unikalnych etykiet
    """
    unique_annotations = {}

    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]

    for record_name in files:
        record_path = os.path.join(directory, record_name)
        try:
            annotation = wfdb.rdann(record_path, 'atr')  # Pobranie adnotacji

            for label in annotation.symbol:
                if label in unique_annotations:
                    unique_annotations[label] += 1
                else:
                    unique_annotations[label] = 1

        except Exception as e:
            print(f"Błąd podczas przetwarzania {record_name}: {e}")

    return unique_annotations



def create_label_map(directory="data/raw/mitdb/"):
    """
    Tworzy dynamiczną mapę etykiet dla adnotacji EKG.
    """
    annotations = extract_unique_annotations(directory)
    sorted_labels = sorted(annotations.keys())  # Sortujemy etykiety
    return {label: idx for idx, label in enumerate(sorted_labels)}