import os
import numpy as np
import wfdb
from scipy.interpolate import interp1d, CubicSpline
from data.qrs_detection import detect_qrs_biosppy
from config import NUM_SAMPLES
from utils.preprocessing import standarize_signal




def interpolate_segment(segment, annotations, segment_start, segment_end, num_samples=NUM_SAMPLES):
    """
    Interpoluje segment EKG do stałej liczby próbek, zachowując poprawne położenie adnotacji.

    :param segment: Oryginalny fragment sygnału EKG (numpy array)
    :param annotations: Lista adnotacji w oryginalnym segmencie (lista indeksów)
    :param segment_start: Indeks początku segmentu w oryginalnym sygnale
    :param segment_end: Indeks końca segmentu w oryginalnym sygnale
    :param num_samples: Docelowa liczba próbek w segmencie
    :return: Interpolowany segment, nowe pozycje adnotacji
    """
    if len(segment) == num_samples:
        return segment, np.array(annotations, dtype=int)

    # Interpolacja sygnału cubic splines
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, num_samples)
    spline = CubicSpline(x_old, segment, extrapolate=True)
    segment_resized = spline(x_new)

    # Skalowanie adnotacji do nowej długości
    scale_factor = num_samples / len(segment)
    new_annotations = np.array([
        round((ann - segment_start) * scale_factor) for ann in annotations
        if segment_start <= ann < segment_end
    ])

    # Usunięcie adnotacji, które wyszły poza zakres segmentu
    new_annotations = new_annotations[(new_annotations >= 0) & (new_annotations < num_samples)]

    return segment_resized, new_annotations




def load_mimic_ecg_segment(patient_id, study_id, directory="data/raw/mimic-iv-ecg"):
    """
    Wczytuje pojedynczy zapis EKG z MIMIC-IV-ECG i mapuje adnotacje na MIT-BIH.

    :param patient_id: ID pacjenta
    :param study_id: ID badania
    :param directory: Ścieżka do katalogu bazy
    :return: (sygnał, adnotacje, etykiety (MIT-BIH format), częstotliwość próbkowania)
    """
    patient_folder = f"p{patient_id}"
    segment_path = os.path.join(directory, patient_folder, study_id)

    record = wfdb.rdrecord(segment_path)
    annotation = wfdb.rdann(segment_path, "atr")

    # Pobranie sygnału i obsługa wielokanałowego EKG (wybór pierwszego kanału)
    signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    signal = standarize_signal(signal)
    fs = record.fs  # Częstotliwość próbkowania


    # Jeśli brak adnotacji, zwracamy pustą tablicę NumPy
    if len(annotation.sample) == 0:
        return signal, np.array([], dtype=int), [], fs

    # Pobranie i mapowanie adnotacji
    beat_annotations = [
        (ann, MIMIC_TO_MITBIH[label]) for ann, label in zip(annotation.sample, annotation.symbol) if label in MIMIC_TO_MITBIH
    ]

    if not beat_annotations:
        return signal, np.array([], dtype=int), [], fs  # Jeśli brak pobudzeń

    # Rozpakowanie pozycji i etykiet pobudzeń
    beat_positions, beat_labels = zip(*beat_annotations)

    return signal, np.array(beat_positions, dtype=int), list(beat_labels), fs



def load_mitbih_segment(record_name, directory="data/raw/mitdb"):
    """
    Loads a single ECG segment from MIT-BIH.

    :param record_name: Record name (e.g., "100")
    :param directory: Path to MIT-BIH dataset
    :return: (ECG signal, annotation positions, annotation labels, sampling frequency)
    """
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    signal = standarize_signal(signal)
    fs = record.fs

    # Obsługa przypadku, gdy nie ma żadnych adnotacji
    if len(annotation.sample) == 0:
        return signal, np.array([], dtype=int), [], fs

    annotations = np.array(annotation.sample, dtype=int)
    labels = annotation.symbol  # Annotation symbols

    return signal, annotations, labels, fs


def load_icentia_segment(patient_id, segment_id, directory="data/raw/icentia11k"):
    """
    Loads a single ECG segment from Icentia11k with only beat annotations (mapped to MIT-BIH format).

    :param patient_id: Patient ID (e.g., 9000)
    :param segment_id: Segment number (e.g., 0, 1, 2, ...)
    :param directory: Path to Icentia11k dataset
    :return: (ECG signal, annotation positions, annotation labels (MIT-BIH format), sampling frequency)
    """
    # Konstrukcja ścieżki do pliku
    patient_folder = f"p0{str(patient_id)[:1]}/p{patient_id:05d}"
    segment_file = f"p{patient_id:05d}_s{segment_id:02d}"
    segment_path = os.path.join(directory, patient_folder, segment_file)

    # Wczytanie rekordu i adnotacji
    record = wfdb.rdrecord(segment_path)
    annotation = wfdb.rdann(segment_path, "atr")

    # Pobranie sygnału EKG (Icentia11k ma 1 kanał)
    signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
    signal = standarize_signal(signal)
    fs = record.fs  # Częstotliwość próbkowania

    # Pozostawienie tylko adnotacji pobudzeń ("N", "S", "V", "Q")
    valid_symbols = {"N", "S", "V", "Q"}
    beat_annotations = [
        (ann, "A" if label == "S" else label)  # Mapowanie "S" → "A"
        for ann, label in zip(annotation.sample, annotation.symbol)
        if label in valid_symbols  # Pomijamy rytmy serca
    ]

    # Jeśli brak pobudzeń, zwracamy pustą listę adnotacji
    if not beat_annotations:
        return signal, np.array([], dtype=int), [], fs  # ⬅ Poprawione zwracanie pustej tablicy numpy


        # Rozpakowanie pozycji i etykiet pobudzeń
    beat_positions, beat_labels = zip(*beat_annotations)

    return signal, np.array(beat_positions, dtype=int), list(beat_labels), fs




def segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples=NUM_SAMPLES):
    """
    Segmentuje EKG na podstawie QRS → QRS z interpolacją i przeskalowaniem adnotacji.

    :param signal: Sygnał EKG (numpy array)
    :param annotations: Pozycje oznaczeń w globalnym sygnale
    :param labels: Symbole oznaczeń (lista stringów, już poprawione w load_mitbih_segment i load_icentia_segment)
    :param qrs_peaks: Pozycje QRS w globalnym sygnale (numpy array)
    :param num_samples: Docelowa liczba próbek w każdym segmencie
    :return: (Tablica segmentów, Tablica etykiet)
    """
    segments = []
    segment_labels_list = []

    for i in range(len(qrs_peaks) - 1):
        start = qrs_peaks[i]
        end = qrs_peaks[i + 1]

        segment = signal[start:end]

        # **Interpolacja segmentu + przeskalowanie adnotacji**
        segment_resized, new_annotations = interpolate_segment(
            segment,
            [ann for ann in annotations if start <= ann < end],
            start,
            end,
            num_samples
        )

        # **Usunięcie segmentów bez adnotacji (unikamy błędnej klasyfikacji)**
        segment_events = [(ann, label) for ann, label in zip(new_annotations, labels) if start <= ann < end]

        if not segment_events:  # Jeśli brak pobudzeń, pomijamy segment
            continue

            # **Zbieranie etykiet segmentu**
        segment_labels = [label for _, label in segment_events]

        segments.append(segment_resized)
        segment_labels_list.append(segment_labels)

    return np.array(segments), np.array(segment_labels_list, dtype=object)  # dtype=object


def load_dataset_records(directory, dataset):
    """
    Pobiera listę rekordów dla podanej bazy danych.

    :param directory: Katalog główny baz danych.
    :param dataset: Nazwa bazy danych ("mitdb", "icentia11k").
    :return: Lista (record_id, segment_id) dla każdego rekordu w bazie.
    """
    dataset_path = os.path.join(directory, dataset)

    if dataset == "mitdb":
        files = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.dat')]
        return [(record_name, None) for record_name in files]  # segment_id = None dla MIT-BIH

    elif dataset == "icentia11k":
        patient_folders = [p for p in os.listdir(dataset_path) if p.startswith("p0")]
        records = []
        for patient_folder in patient_folders:
            patient_id = int(patient_folder[2:])
            patient_path = os.path.join(dataset_path, patient_folder, f"p{patient_id:05d}")

            segment_files = [f.split('_s')[1].split('.')[0] for f in os.listdir(patient_path) if f.endswith('.dat')]
            records.extend([(patient_id, int(segment_id)) for segment_id in segment_files])

        return records

    elif dataset == "mimic-iv-ecg":
        patient_folders = [p for p in os.listdir(dataset_path) if p.startswith("p")]
        records = []
        for patient_folder in patient_folders:
            patient_id = int(patient_folder[1:])  # np. "p1000" → 1000
            patient_path = os.path.join(dataset_path, patient_folder)

            study_files = [f.split('.')[0] for f in os.listdir(patient_path) if f.endswith('.dat')]
            records.extend([(patient_id, study_id) for study_id in study_files])


    else:
        print(f"⚠️ Dataset {dataset} is not supported!")
        return []


def process_record(dataset, record_id, segment_id, directory, num_samples=NUM_SAMPLES):
    """
    Wczytuje, wykrywa QRS i segmentuje pojedynczy rekord.

    :param dataset: Nazwa bazy danych ("mitdb", "icentia11k").
    :param record_id: ID rekordu (lub ID pacjenta dla Icentia).
    :param segment_id: ID segmentu (dla Icentia) lub None dla MIT-BIH.
    :param directory: Ścieżka do katalogu baz danych.
    :param num_samples: Liczba próbek na segment.
    :return: (Segmentowane dane, Etykiety one-hot)
    """
    dataset_loaders = {
        "mitdb": load_mitbih_segment,
        "icentia11k": load_icentia_segment,
        "mimic-iv-ecg": load_mimic_ecg_segment
    }

    # Wczytaj dane
    if dataset == "mitdb":
        signal, annotations, labels, fs = dataset_loaders[dataset](record_id, directory)
    else:
        signal, annotations, labels, fs = dataset_loaders[dataset](record_id, segment_id, directory)

    # Wykryj QRS
    qrs_peaks = detect_qrs_biosppy(signal, fs)

    # Segmentuj dane
    return segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, num_samples)




def prepare_qrs_dataset(directory="data/raw/", datasets=["mitdb", "icentia11k"], num_samples=NUM_SAMPLES):
    """
    Ładuje i segmentuje rekordy z wielu baz danych.

    :param directory: Katalog główny.
    :param datasets: Lista nazw baz danych do wczytania.
    :param num_samples: Liczba próbek w segmencie.
    :return: (Wszystkie segmenty, Etykiety one-hot)
    """
    all_segments = []
    all_labels = []

    for dataset in datasets:
        records = load_dataset_records(directory, dataset)

        for record_id, segment_id in records:
            X, y = process_record(dataset, record_id, segment_id, directory, num_samples)
            all_segments.append(X)
            all_labels.append(y)

    return np.concatenate(all_segments), np.concatenate(all_labels)








MIMIC_TO_MITBIH = {
    "N": "N",  # Normalny rytm
    "L": "L",  # LBBB
    "R": "R",  # RBBB
    "A": "A",  # Pobudzenie przedsionkowe
    "a": "A",  # Aberrated atrial beat
    "J": "J",  # Pobudzenie węzłowe
    "S": "A",  # Pobudzenie nadkomorowe (PAC)
    "V": "V",  # PVC - Pobudzenie komorowe
    "F": "F",  # Fusion beat
    "Q": "Q"   # Niezidentyfikowane pobudzenie
}



LABEL_MAP = {
    "N": 0,   # Normal
    "L": 1,   # Left bundle branch block
    "R": 2,   # Right bundle branch block
    "A": 3,   # Atrial premature beat
    "a": 4,   # Aberrated atrial premature beat
    "J": 5,   # Junctional premature beat
    "S": 6,   # Supraventricular premature beat
    "V": 7,   # Ventricular premature beat
    "r": 8,   # R-on-T premature ventricular contraction
    "F": 9,   # Fusion of ventricular and normal beat
    "e": 10,  # Atrial escape beat
    "j": 11,  # Junctional escape beat
    "n": 12,  # Supraventricular escape beat
    "E": 13,  # Ventricular escape beat
    "/": 14,  # Paced beat
    "f": 15,  # Fusion of paced and normal beat
    "Q": 16   # Unclassifiable beat
}


# Mapowanie Icentia11k na MIT-BIH
ICENTIA_TO_MITBIH = {
    "N": "N",  # Normal → Normal
    "S": "A",  # PAC → Atrial Premature Beat
    "V": "V",  # PVC → Ventricular Premature Beat
    "Q": "Q"   # Unknown → Unclassifiable
}

