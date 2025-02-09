import os
import wfdb
import numpy as np



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






# **Funkcje przetwarzania sygnału**
def standarize_signal(ecg_signal):
    """Standaryzacja Z-score"""
    mean = np.mean(ecg_signal)
    std = np.std(ecg_signal)
    return (ecg_signal - mean) / std if std != 0 else ecg_signal

def normalize_signal(ecg_signal):
    """Normalizacja Min-Max do przedziału [0,1]"""
    min_val = np.min(ecg_signal)
    max_val = np.max(ecg_signal)
    return (ecg_signal - min_val) / (max_val - min_val) if max_val != min_val else ecg_signal

def filter_rare_classes(X, Y, min_samples=5):
    """Usuwa klasy, które mają mniej niż `min_samples` próbek."""
    unique, counts = np.unique(Y, return_counts=True)
    class_counts = dict(zip(unique, counts))  # Liczba wystąpień każdej klasy

    # Klasy, które mają co najmniej `min_samples`
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]

    print(f"📊 Liczba próbek w każdej klasie przed filtracją: {class_counts}")
    print(f"🎯 Klasy po filtracji: {valid_classes}")

    # Jeśli po filtracji nie zostały żadne klasy → rzuć błąd
    if not valid_classes:
        raise ValueError("❌ Wszystkie klasy zostały odrzucone przez filtrację! Zmniejsz `min_samples`.")

    # Tworzenie maski dla segmentów zawierających przynajmniej jedną ważną klasę
    mask = np.isin(Y, valid_classes)

    X_filtered = X[mask]
    Y_filtered = Y[mask]  # Nie trzeba indeksować drugiego wymiaru, bo `Y` jest 1D

    print(f"✅ Po filtracji: X.shape={X_filtered.shape}, Y.shape={Y_filtered.shape}")
    return X_filtered, Y_filtered



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



