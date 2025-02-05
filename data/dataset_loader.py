import os
import numpy as np
import wfdb
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks

from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import create_label_map, standarize_signal,normalize_signal
from config import NUM_SAMPLES  # Importujemy stałą
from utils.visualization import visualize_interpolation
from scipy.signal import resample




def load_ecg_record(record_name, directory="data/raw/mitdb/"):
    """
    Wczytuje EKG i adnotacje z MIT-BIH.
    """
    record_path = directory + record_name
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')  # Pobranie adnotacji

    return record.p_signal, annotation.sample, annotation.symbol, record.fs



def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples=NUM_SAMPLES):
    """
    Interpoluje segment EKG do stałej liczby próbek, używając cubic splines,
    jednocześnie zachowując dokładne położenie adnotacji.

    :param segment: Oryginalny fragment sygnału EKG (numpy array)
    :param annotations: Lista adnotacji w oryginalnym segmencie (lista indeksów)
    :param segment_start: Indeks początku segmentu w oryginalnym sygnale
    :param segment_end: Indeks końca segmentu w oryginalnym sygnale
    :param num_samples: Docelowa liczba próbek w segmencie
    :return: Interpolowany segment, nowe pozycje adnotacji
    """
    if len(segment) == num_samples:
        return segment, np.array(annotations, dtype=int)

    # **Interpolacja sygnału cubic splines**
    x_old = np.linspace(0, 1, len(segment))  # Oryginalna siatka czasowa
    x_new = np.linspace(0, 1, num_samples)  # Nowa siatka czasowa
    spline = CubicSpline(x_old, segment, extrapolate=True)
    segment_resized = spline(x_new)

    # **Skalowanie adnotacji do nowej długości**
    scale_factor = num_samples / len(segment)
    new_annotations = [
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ]

    # **Zabezpieczenie przed błędami indeksowania**
    new_annotations = np.clip(new_annotations, 0, num_samples - 1)

    # 🟢 Wizualizacja (jeśli chcesz zobaczyć efekt interpolacji)
    #visualize_interpolation(segment_resized, new_annotations)

    return segment_resized, np.array(new_annotations, dtype=int)










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
    """ Wczytuje pliki, segmentuje według QRS i przygotowuje zbiór do trenowania CNN. """
    all_segments, all_labels = [], []
    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]


    for record_name in files:
        record_path = os.path.join(directory, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')


        signal = standarize_signal(record.p_signal[:, 0])
        annotations, labels = annotation.sample, annotation.symbol
        qrs_peaks = detect_qrs_biosppy(signal, record.fs)
        X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples)

        all_segments.append(X)
        all_labels.append(y)

    return np.concatenate(all_segments), np.concatenate(all_labels)


ICENTIA_TO_MIT_BIH = {
    "N": "N",   # Normalne pobudzenie → Normalne pobudzenie
    "S": "A",   # ESSV (PAC) → Pobudzenie przedsionkowe (Atrial)
    "V": "V",   # PVC → PVC (Przedwczesne pobudzenie komorowe)
    "Q": "Q",   # Nieznane pobudzenie → Nieznane pobudzenie
}

