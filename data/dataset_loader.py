import numpy as np
import os
import pickle
import wfdb
from scipy.interpolate import interp1d
from data.process_data import save_processed_data, load_processed_data
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal, add_medical_noise
from config import LABEL_MAP, NUM_SAMPLES
from utils.visualization import visualize_interpolation


def load_ecg_record(record_name, directory="data/raw/mitdb/"):
    """Wczytuje EKG i adnotacje z MIT-BIH."""
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    print(f"✅ Wczytano {record_name}: sygnał={record.p_signal.shape}, adnotacje={len(annotation.sample)}")
    return record.p_signal, annotation.sample, annotation.symbol, record.fs

def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples):
    """Interpoluje segment EKG do stałej liczby próbek."""
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, num_samples)
    spline = interp1d(x_old, segment, kind="linear", fill_value="extrapolate")
    segment_resized = spline(x_new)

    segment_noisy = add_medical_noise(segment_resized)

    scale_factor = num_samples / len(segment)
    new_annotations = [
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ]
    new_annotations = np.clip(new_annotations, 0, num_samples - 1)


    return segment_noisy, np.array(new_annotations, dtype=int)


def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples):
    """
    Segmentuje EKG na podstawie QRS → QRS i przypisuje etykiety.
    - Interpoluje **tylko jeśli są anomalie**.
    - Usuwa segmenty z **tylko normalnym rytmem (`0`)**.
    """
    segments = []
    segment_labels = []

    print("📊 Rozpoczęcie segmentacji...")

    for i in range(len(qrs_peaks) - 1):
        start, end = qrs_peaks[i], qrs_peaks[i + 1]
        segment = signal[start:end]

        # 🔹 Pobranie etykiet z segmentu
        segment_events = [
            (ann, label) for ann, label in zip(annotations, labels)
            if start <= ann < end and label in LABEL_MAP
        ]

        # **1️⃣ Sprawdzenie, czy są anomalie**
        if segment_events:
            segment_label = list(set(LABEL_MAP[label] for _, label in segment_events))
            segment_label = [lbl for lbl in segment_label if lbl != 0]  # Usuń normalny rytm

            if not segment_label:
                continue  # 🔥 Ignorujemy jeśli nic nie zostało

            # 🔥 Interpolacja tylko jeśli segment zawiera anomalie
            segment_resized, new_annotations = interpolate_segment(
                segment, [ann for ann, _ in segment_events], start, end, num_samples
            )

            # 🔥 Wizualizacja tylko jeśli segment był interpolowany
            #visualize_interpolation(segment_resized, new_annotations, segment_id=i)

        else:
            continue  # 🔥 Ignorujemy segmenty bez anomalii

        # 🔹 Wybór **głównej klasy** (zamiast multi-label)
        primary_label = segment_label[0]

        segments.append(segment_resized)
        segment_labels.append(primary_label)

        print(f"✅ Segment {i}: kształt={segment_resized.shape}, etykieta={primary_label}, nowe adnotacje={new_annotations}")

    print(f"✅ Segmentacja zakończona: {len(segments)} segmentów (bez normalnych rytmów)")
    return np.array(segments), np.array(segment_labels)



def prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES):
    """
    Wczytuje pliki, segmentuje według QRS i przygotowuje zbiór do trenowania CNN.
    """
    filename = "ekg_segments"

    print("📥 Przetwarzanie danych od zera...")

    all_segments, all_labels = [], []
    files = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.dat')]

    for record_name in files:
        signal, annotations, labels, fs = load_ecg_record(record_name, directory)
        signal = standarize_signal(signal[:, 0])
        qrs_peaks = detect_qrs_biosppy(signal, fs)

        X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples)
        all_segments.extend(X)
        all_labels.extend(y)

    # Zapisanie przetworzonych danych
    save_processed_data(np.array(all_segments), np.array(all_labels), filename)

    print(f"✅ Zapisano dane: {len(all_segments)} próbek, {len(set(all_labels))} unikalnych klas")
    return np.array(all_segments), np.array(all_labels)
