import numpy as np
import os
import wfdb
from scipy.signal import resample
from collections import Counter
from data.process_data import save_processed_data
from utils.preprocessing import standarize_signal
from config import LABEL_MAP, NUM_SAMPLES, MITDB_PATH, SVDB_PATH


def load_and_resample_ecg(record_name, directory, target_fs=360):
    """🔍 Wczytuje EKG i adnotacje z SVDB, resampluje cały sygnał do `target_fs`, a następnie przesuwa adnotacje."""
    record_path = os.path.join(directory, record_name)

    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
    except Exception as e:
        print(f"❌ Błąd wczytywania pliku {record_name}: {e}")
        return None, None, None, None

    original_fs = record.fs  # Oryginalna częstotliwość próbkowania (np. 128 Hz dla SVDB)
    signal = record.p_signal[:, 0]  # Wybieramy pierwszy kanał EKG

    if original_fs != target_fs:
        # Obliczamy nową liczbę próbek
        new_length = int(len(signal) * (target_fs / original_fs))
        signal = resample(signal, new_length)  # Resamplowanie całego sygnału

        # Przesunięcie indeksów adnotacji
        annotation.sample = (annotation.sample * (target_fs / original_fs)).astype(int)

    return signal, annotation.sample, annotation.symbol, target_fs



def segment_around_qrs(signal, qrs_peaks, labels, target_samples=300):
    """🔍 Segmentuje EKG wokół QRS (środek = R-peak), przypisując klasę z QRS."""
    segments, segment_labels = [], []
    median_rr = int(np.median(np.diff(qrs_peaks))) if len(qrs_peaks) > 1 else 300

    for i, r in enumerate(qrs_peaks):
        window_size = int(median_rr // 2)
        start, end = int(max(0, r - window_size)), int(min(len(signal), r + window_size))

        segment = signal[start:end]

        # 🔹 **Naprawa błędu** – wyrównanie długości segmentów
        if len(segment) < target_samples:
            segment = np.pad(segment, (0, target_samples - len(segment)), mode='constant')
        elif len(segment) > target_samples:
            segment = segment[:target_samples]

        # Pobranie klasy
        primary_label = labels[i] if i < len(labels) else None
        if primary_label not in LABEL_MAP:
            continue

        segments.append(segment)
        segment_labels.append(LABEL_MAP[primary_label])

    # 🔹 **Naprawa błędu** – konwersja do `np.array()` z jednolitą długością
    return np.array(segments, dtype=np.float32), np.array(segment_labels, dtype=np.int32)



def prepare_qrs_dataset(directory, target_fs=360):
    """🔍 Wczytuje pliki, resampluje SVDB do `target_fs`, segmentuje wokół QRS i zapisuje przetworzone dane."""
    all_segments, all_labels = [], []
    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]

    print(f"📂 Znaleziono {len(files)} plików EKG w katalogu {directory}")

    for record_name in files:
        signal, qrs_peaks, labels, fs = load_and_resample_ecg(record_name, directory, target_fs)
        if signal is None:
            continue

        print(f"🔍 Przetwarzanie rekordu {record_name}, liczba QRS: {len(qrs_peaks)}")

        signal = standarize_signal(signal)  # Standaryzacja sygnału

        # Segmentacja wokół QRS
        X, y = segment_around_qrs(signal, qrs_peaks, labels)

        if len(X) == 0:
            continue

        all_segments.extend(X)
        all_labels.extend(y)

    # Konwersja do NumPy
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)

    print(f"✅ Finalna liczba segmentów: {len(all_segments)}")
    print(f"📊 Liczność klas przed zapisaniem: {Counter(all_labels)}")

    if len(all_segments) == 0:
        print("❌ Brak segmentów do zapisania! Sprawdź dane wejściowe.")
        return None, None

    # Zapisywanie przetworzonych danych
    save_processed_data(all_segments, all_labels, "ekg_segments")

    return all_segments, all_labels
