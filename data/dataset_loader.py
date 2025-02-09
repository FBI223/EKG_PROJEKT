import numpy as np
import os
import pickle
import wfdb
from scipy.interpolate import interp1d
from data.process_data import save_processed_data, load_processed_data
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal, add_medical_noise, select_highest_priority_class
from config import LABEL_MAP, NUM_SAMPLES, MITDB_PATH
from utils.visualization import visualize_interpolation, visualize_plot_interpolation
from scipy.signal import resample



def load_ecg_record(record_name, directory=MITDB_PATH):
    """Wczytuje EKG i adnotacje z MIT-BIH."""
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    return record.p_signal, annotation.sample, annotation.symbol, record.fs




def process_segment(segment, annotations, segment_start, segment_end):
    """Przetwarza segment EKG, zapewniając dokładnie `NUM_SAMPLES` próbek za pomocą `resample`."""


    original_length = len(segment)

    # Sprawdzenie, czy brakuje więcej niż 1/2 próbek
    if original_length < NUM_SAMPLES * (1/2):
        return None, None  # Odrzucenie segmentu


    # 1️⃣ **Dodanie szumu (jeśli jest potrzebne)**
    segment_noisy = add_medical_noise(segment)
    segment_resampled = resample(segment_noisy, NUM_SAMPLES)


    # 3️⃣ **Skalowanie anotacji**
    scale_factor = NUM_SAMPLES / original_length
    new_annotations = [
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ]
    new_annotations = np.clip(new_annotations, 0, NUM_SAMPLES - 1)

    return segment_resampled, np.array(new_annotations, dtype=int)



def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, patient_name="unknown"):
    """
    Segmentuje EKG na podstawie QRS → QRS i przypisuje etykiety.
    - **Używa `resample` tylko na końcu**.
    - **Nie sprawdza ponownie liczby próbek** (bo `resample` już to robi).
    """
    segments = []
    segment_labels = []

    for i in range(len(qrs_peaks) - 1):
        start, end = qrs_peaks[i], qrs_peaks[i + 1]
        segment = signal[start:end]

        # 🔹 Pobranie etykiet z segmentu
        segment_events = [
            (ann, label) for ann, label in zip(annotations, labels)
            if start <= ann < end and label in LABEL_MAP
        ]

        # 🔹 Przetwarzanie segmentu (z `resample` na końcu)
        segment_resized, new_annotations = process_segment(
            segment, [ann for ann, _ in segment_events], start, end
        )


        # **Jeśli segment został odrzucony – pomijamy**
        if segment_resized is None:
            continue

            # ✅ Wizualizacja segmentu
        #visualize_interpolation(segment_resized, newz_annotations, segment_id=i, patient_name=patient_name)

        # **Jeśli są etykiety, wybieramy je, w przeciwnym razie przypisujemy `0` (normalny rytm)**
        segment_label = list(set(LABEL_MAP[label] for _, label in segment_events)) if segment_events else [0]

        # 🔹 Wybór głównej klasy – jeśli istnieje inna niż `0`, to ją wybieramy
        primary_label = select_highest_priority_class(segment_label)

        #print(segment_label)
        #print(primary_label)
        #print()

        segments.append(segment_resized)
        segment_labels.append(primary_label)

    return np.array(segments), np.array(segment_labels)


def prepare_qrs_dataset(directory=MITDB_PATH):
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

        # 🔥 Rysowanie pełnego sygnału EKG i zapis do pliku
        #plot_full_ecg_record(record_name, signal, annotations, labels, fs)

        # 🔥 Segmentacja sygnału
        X, y = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, record_name)
        all_segments.extend(X)
        all_labels.extend(y)

    # ✅ Konwersja listy do tablicy NumPy
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)

    # 🔥 Zapisanie przetworzonych danych
    save_processed_data(all_segments, all_labels, filename)

    print(f"✅ Zapisano dane: {len(all_segments)} próbek, {len(set(all_labels))} unikalnych klas")
    return all_segments, all_labels
