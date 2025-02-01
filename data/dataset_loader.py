import wfdb
import numpy as np
import os

def load_ecg_record(record_name, directory='data/raw/mitdb/'):
    """
    Wczytuje pojedynczy zapis EKG z lokalnej bazy.
    :param record_name: Nazwa pliku (np. '100' dla MIT-BIH)
    :param directory: Ścieżka do lokalnej bazy
    :return: sygnał EKG (NumPy array), adnotacje (jeśli dostępne), częstotliwość próbkowania
    """
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')  # Adnotacje lekarzy (arytmie)

    return record.p_signal, annotation.sample, record.fs  # Sygnał, pozycje anomalii, częstotliwość

def load_full_database(directory='data/raw/mitdb/'):
    """
    Pobiera i wczytuje wszystkie dostępne zapisy EKG z lokalnej bazy.
    :param directory: Ścieżka do lokalnej bazy
    :return: Słownik z sygnałami EKG
    """
    records = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]
    data = {}

    for record in records:
        print(f"Wczytywanie rekordu: {record}")
        try:
            signal, annotations, fs = load_ecg_record(record, directory)
            data[record] = {'signal': signal, 'annotations': annotations, 'fs': fs}
        except Exception as e:
            print(f"Błąd przy wczytywaniu {record}: {e}")

    return data
