import os
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from utils.preprocessing import normalize_signal, create_label_map
from config import NUM_SAMPLES  # Importujemy stałą



def load_ecg_record(record_name, directory="data/raw/mitdb/"):
    """
    Wczytuje EKG i adnotacje z MIT-BIH.
    """
    record_path = directory + record_name
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')  # Pobranie adnotacji

    return record.p_signal, annotation.sample, annotation.symbol, record.fs


def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples):
    """
    Interpoluje segment EKG do stałej liczby próbek, jednocześnie zachowując dokładne położenie adnotacji.

    :param segment: Oryginalny fragment sygnału EKG (numpy array)
    :param annotations: Lista adnotacji w oryginalnym segmencie (lista indeksów)
    :param segment_start: Indeks początku segmentu w oryginalnym sygnale
    :param segment_end: Indeks końca segmentu w oryginalnym sygnale
    :param num_samples: Docelowa liczba próbek w segmencie
    :return: Interpolowany segment, nowe pozycje adnotacji
    """
    if len(segment) == num_samples:
        return segment, np.array(annotations, dtype=int)  # Jeśli długość już jest prawidłowa, nic nie zmieniamy

    # **Interpolacja sygnału**
    x_old = np.linspace(0, 1, len(segment))  # Oryginalna siatka czasowa
    x_new = np.linspace(0, 1, num_samples)  # Nowa siatka czasowa
    interpolator = interp1d(x_old, segment, kind='linear', fill_value="extrapolate")
    segment_resized = interpolator(x_new)

    # **Skalowanie adnotacji do nowej długości**
    new_annotations = []
    for ann in annotations:
        if segment_start <= ann < segment_end:
            rel_pos = (ann - segment_start) / (segment_end - segment_start)  # Pozycja w zakresie [0,1]
            new_index = round(rel_pos * (num_samples - 1))  # 🟢 Poprawiona linia
            new_annotations.append(new_index)

    # **Zabezpieczenie przed błędami indeksowania**
    new_annotations = np.clip(new_annotations, 0, num_samples - 1)

    return segment_resized, np.array(new_annotations, dtype=int)


def detect_qrs(signal, fs):
    """
    Wykrywa zespoły QRS jako punkty odniesienia do segmentacji cykli serca.
    """
    peaks, _ = find_peaks(signal, height=0.5, distance=fs * 0.6)  # Minimalny odstęp 600ms
    return peaks


def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples=NUM_SAMPLES):
    """
    Segmentuje EKG na podstawie QRS → QRS i przeskalowuje do stałej długości.

    :param signal: Sygnał EKG
    :param annotations: Pozycje oznaczeń
    :param labels: Symbole oznaczeń
    :param qrs_peaks: Pozycje załamków QRS
    :param num_samples: Docelowa liczba próbek w każdym cyklu (stała długość)
    :return: Tablica segmentów, etykiety w postaci listy indeksów klas
    """
    segments = []
    segment_labels = []

    label_map = create_label_map("data/raw/mitdb/")

    for i in range(len(qrs_peaks) - 1):
        start = qrs_peaks[i]
        end = qrs_peaks[i + 1]

        segment = signal[start:end]

        # Pobranie adnotacji w segmencie (ignorujemy nieznane)
        segment_events = [
            (ann, label) for ann, label in zip(annotations, labels)
            if start <= ann < end and label in label_map
        ]

        segment_resized, new_annotations = interpolate_segment(
            segment, [ann for ann, _ in segment_events], start, end, num_samples
        )


        # Jeśli są adnotacje, wybieramy dominującą klasę, jeśli nie, przypisujemy "N" (normalny rytm)
        if segment_events:
            segment_label = max(set([label_map[label] for _, label in segment_events]),
                                key=[label_map[label] for _, label in segment_events].count)
        else:
            segment_label = label_map['N']

        segments.append(segment_resized)
        segment_labels.append(segment_label)

    return np.array(segments), np.array(segment_labels)





def prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES):
    """
    Wczytuje wszystkie pliki, segmentuje według QRS i przygotowuje zbiór do trenowania CNN (bez one-hot encoding).

    :param directory: Ścieżka do bazy MIT-BIH.
    :param num_samples: Docelowa liczba próbek w każdym segmencie.
    :return: X (sygnały EKG), y (etykiety numeryczne)
    """
    all_segments = []
    all_labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            record_name = filename.split('.')[0]
            record = wfdb.rdrecord(os.path.join(directory, record_name))
            annotation = wfdb.rdann(os.path.join(directory, record_name), 'atr')

            signal = record.p_signal[:, 0]  # Używamy tylko pierwszego kanału
            signal = normalize_signal(signal)

            annotations = annotation.sample
            labels = annotation.symbol

            # **Wykrywanie QRS**
            qrs_peaks = detect_qrs(signal, record.fs)

            # **Segmentacja według QRS**
            X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, NUM_SAMPLES)

            all_segments.append(X)
            all_labels.append(y)

    return np.concatenate(all_segments), np.concatenate(all_labels)



