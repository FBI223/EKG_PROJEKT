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