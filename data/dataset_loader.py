import numpy as np
import os
import wfdb
from scipy.signal import resample
from data.process_data import save_processed_data
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal, add_medical_noise, select_highest_priority_class
from config import LABEL_MAP, NUM_SAMPLES, MITDB_PATH

def load_ecg_record(record_name, directory=MITDB_PATH):
    """Wczytuje EKG i adnotacje z MIT-BIH."""
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    return record.p_signal, annotation.sample, annotation.symbol, record.fs

def process_segment(segment, annotations, segment_start, segment_end):
    """Przetwarza segment EKG, zapewniając dokładnie `NUM_SAMPLES` próbek za pomocą `resample`."""
    original_length = len(segment)

    # Odrzucanie segmentów, które są zbyt krótkie
    if original_length < NUM_SAMPLES * 0.5:
        return None, None

        # Dodanie szumu i resampling
    segment_noisy = add_medical_noise(segment)
    segment_resampled = resample(segment_noisy, NUM_SAMPLES)

    # Skalowanie anotacji do nowego rozmiaru segmentu
    scale_factor = NUM_SAMPLES / original_length
    new_annotations = [
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ]
    new_annotations = np.clip(new_annotations, 0, NUM_SAMPLES - 1)

    return segment_resampled, np.array(new_annotations, dtype=int)

def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks):
    """Segmentuje EKG na podstawie wykrytych QRS-ów i przypisuje etykiety."""
    segments, segment_labels = [], []

    for i in range(len(qrs_peaks) - 1):
        start, end = qrs_peaks[i], qrs_peaks[i + 1]
        segment = signal[start:end]

        # Pobranie etykiet dla segmentu
        segment_events = [
            (ann, label) for ann, label in zip(annotations, labels)
            if start <= ann < end and label in LABEL_MAP
        ]

        # Przetwarzanie segmentu (resampling i filtracja)
        segment_resized, new_annotations = process_segment(
            segment, [ann for ann, _ in segment_events], start, end
        )

        # Pomijanie segmentów, które nie spełniły warunków
        if segment_resized is None or len(segment_events) == 0:
            continue

        # Wybór klasy dla segmentu (priorytetyzacja)
        segment_label = [LABEL_MAP[label] for _, label in segment_events]
        primary_label = select_highest_priority_class(segment_label)

        segments.append(segment_resized)
        segment_labels.append(primary_label)

    return np.array(segments), np.array(segment_labels)

def prepare_qrs_dataset(directory=MITDB_PATH):
    """Wczytuje pliki, segmentuje według QRS i przygotowuje zbiór do trenowania CNN."""
    print("📥 Przetwarzanie danych od zera...")

    all_segments, all_labels = [], []
    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]

    for record_name in files:
        signal, annotations, labels, fs = load_ecg_record(record_name, directory)
        signal = standarize_signal(signal[:, 0])
        qrs_peaks = detect_qrs_biosppy(signal, fs)

        # Segmentacja sygnału
        X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks)

        all_segments.extend(X)
        all_labels.extend(y)

    # Konwersja do NumPy
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)

    # 🔥 Debugging: Weryfikacja liczby klas przed oversamplingiem
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"🔍 Liczba próbek przed oversamplingiem: {len(all_segments)}")
    print(f"🔍 Klasy przed oversamplingiem: {dict(zip(unique_labels, counts))}")

    # Zapisanie przetworzonych danych
    save_processed_data(all_segments, all_labels, "ekg_segments")

    print(f"✅ Zapisano dane: {len(all_segments)} próbek, {len(set(all_labels))} unikalnych klas")
    return all_segments, all_labels
