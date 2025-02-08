import os
import numpy as np
import wfdb
from scipy.interpolate import interp1d, CubicSpline
from sklearn.preprocessing import MultiLabelBinarizer
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import *
from config import NUM_SAMPLES  # Importujemy stałą
from utils.visualization import visualize_interpolation


def load_ecg_record(record_name, directory="data/raw/mitdb/"):
    """
    Wczytuje EKG i adnotacje z MIT-BIH.
    """
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')  # Pobranie adnotacji

    return record.p_signal, annotation.sample, annotation.symbol, record.fs


def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples):
    """
    Interpoluje segment EKG do stałej liczby próbek za pomocą `CubicSpline`,
    a następnie dodaje szum medyczny.

    :param segment: Oryginalny fragment sygnału EKG (numpy array)
    :param annotations: Lista adnotacji w oryginalnym segmencie (lista indeksów)
    :param segment_start: Indeks początku segmentu w oryginalnym sygnale
    :param segment_end: Indeks końca segmentu w oryginalnym sygnale
    :param num_samples: Docelowa liczba próbek w segmencie
    :return: Interpolowany segment z dodanym szumem, nowe pozycje adnotacji
    """
    if len(segment) == num_samples:
        return segment, np.array(annotations, dtype=int)

    # Interpolacja sygnału za pomocą `CubicSpline`
    x_old = np.linspace(0, 1, len(segment))  # Oryginalna siatka czasowa
    x_new = np.linspace(0, 1, num_samples)  # Nowa siatka czasowa
    spline = CubicSpline(x_old, segment, extrapolate=True)
    segment_resized = spline(x_new)

    # Dodanie szumów medycznych po interpolacji
    segment_noisy = add_medical_noise(segment_resized)

    # Skalowanie adnotacji do nowej długości
    scale_factor = num_samples / len(segment)
    new_annotations = [
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ]

    # Zabezpieczenie przed błędami indeksowania
    new_annotations = np.clip(new_annotations, 0, num_samples - 1)

    #visualize_interpolation(segment_noisy, new_annotations)

    return segment_noisy, np.array(new_annotations, dtype=int)

def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples):
    """
    Segmentuje EKG na podstawie QRS → QRS i przeskalowuje do stałej długości.

    :param signal: Sygnał EKG
    :param annotations: Pozycje oznaczeń
    :param labels: Symbole oznaczeń
    :param qrs_peaks: Pozycje załamków QRS
    :param num_samples: Docelowa liczba próbek w każdym cyklu (stała długość)
    :return: Tablica segmentów, etykiety jako lista indeksów klas (multi-label)
    """
    segments = []
    segment_labels = []

    label_map = create_label_map("data/raw/mitdb/")

    for i in range(len(qrs_peaks) - 1):
        start = qrs_peaks[i]
        end = qrs_peaks[i + 1]

        segment = signal[start:end]

        # Pobranie adnotacji w segmencie
        segment_events = [
            (ann, label) for ann, label in zip(annotations, labels)
            if start <= ann < end and label in label_map
        ]

        # 🔵 Prawidłowe wywołanie `interpolate_segment()`
        segment_resized, new_annotations = interpolate_segment(
            segment, [ann for ann, _ in segment_events], start, end, num_samples
        )

        # 🔵 MULTI-LABEL: Zachowujemy wszystkie klasy zamiast tylko jednej
        if segment_events:
            segment_label = list(set(label_map[label] for _, label in segment_events))
        else:
            segment_label = [label_map['N']]  # Jeśli brak etykiet, przypisujemy normalny rytm

        segments.append(segment_resized)
        segment_labels.append(segment_label)

        # ✅ DEBUG: Sprawdź czy segmenty mają właściwe klasy
        #print(f"🔍 Segment {i}: kształt={segment_resized.shape}, etykieta={segment_label}, nowe adnotacje={new_annotations}")

    return np.array(segments), segment_labels  # 🔵 segment_labels to teraz lista list!


def prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES):
    """
    Wczytuje pliki, segmentuje według QRS i przygotowuje zbiór do trenowania CNN.
    Obsługuje multi-label classification.
    """
    all_segments, all_labels = [], []
    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]

    for record_name in files:
        signal, annotations, labels, fs = load_ecg_record(record_name, directory)
        signal = standarize_signal(signal[:, 0])  # Pobranie pierwszego kanału (ECG Lead I)
        qrs_peaks = detect_qrs_biosppy(signal, fs)

        X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples)
        all_segments.extend(X)  # ✅ Spłaszczenie segmentów
        all_labels.extend(y)  # ✅ Lista multi-label

    # **Konwersja etykiet do MultiLabelBinarizer**
    mlb = MultiLabelBinarizer()
    all_labels = mlb.fit_transform(all_labels)

    return np.array(all_segments), np.array(all_labels)





def add_medical_noise(signal, noise_level=0.01, noise_type="impulse"):
    """
    Dodaje szum medyczny do sygnału EKG.

    :param signal: Oryginalny sygnał EKG (numpy array)
    :param noise_level: Poziom szumu (procent wartości maksymalnej sygnału)
    :param noise_type: Typ szumu: "gaussian", "impulse", "pink"
    :return: Sygnał EKG z dodanym szumem
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level * np.max(signal), size=signal.shape)
    elif noise_type == "impulse":
        noise = np.random.choice([0, np.max(signal) * noise_level], size=signal.shape, p=[0.98, 0.02])
    elif noise_type == "pink":
        freqs = np.fft.rfftfreq(len(signal))
        pink_noise = np.random.randn(len(freqs)) / (freqs + 1e-4)
        noise = np.fft.irfft(pink_noise) * noise_level * np.max(signal)
    else:
        raise ValueError("Nieznany typ szumu!")

    signal_noisy = signal + noise
    return np.clip(signal_noisy, np.min(signal), np.max(signal))  # 🔵 Zapobiegamy wartościom ekstremalnym




