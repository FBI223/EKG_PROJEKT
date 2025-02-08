import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

from config import NUM_SAMPLES
from data.dataset_loader import load_ecg_record, segment_ecg_by_qrs
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal


def test_model_on_patient(model, patient_record, directory="data/raw/mitdb/"):
    """
    Testuje model na rzeczywistych danych pacjenta z MIT-BIH.

    :param model: Wytrenowany model CNN
    :param patient_record: Nazwa pliku pacjenta (bez rozszerzenia .dat)
    :param directory: Folder z danymi pacjenta
    """
    print(f"🩺 Testowanie modelu na pacjencie: {patient_record}")

    # **1️⃣ Załaduj dane pacjenta**
    signal, annotations, labels, fs = load_ecg_record(patient_record, directory)
    signal = standarize_signal(signal[:, 0])  # Pobieramy pierwszy kanał EKG
    qrs_peaks = detect_qrs_biosppy(signal, fs)

    # **2️⃣ Segmentacja EKG**
    X_test_patient, y_test_patient = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, NUM_SAMPLES)

    # **3️⃣ Konwersja do multi-label**
    mlb = MultiLabelBinarizer()
    y_test_patient = mlb.fit_transform(y_test_patient)

    # **4️⃣ Modelowanie i predykcja**
    X_test_patient = np.expand_dims(X_test_patient, axis=-1)  # Dopasowanie do modelu CNN
    y_pred = model.predict(X_test_patient)

    # Zamiana sigmoid na 0/1 (próg 0.5)
    y_pred = (y_pred > 0.5).astype(int)

    # **5️⃣ Raport z testowania**
    target_names = [f"Klasa {str(i)}" for i in range(y_pred.shape[1])]  # 👈 Konwersja na stringi
    print("📊 Classification Report dla pacjenta:")
    print(classification_report(y_test_patient, y_pred, target_names=target_names, zero_division=0))  # 👈 zero_division=0

    # **6️⃣ Wizualizacja wyników dla losowych segmentów**
    num_samples_to_plot = min(5, len(X_test_patient))  # Upewnij się, że nie wyjdziemy poza zakres
    random_indices = random.sample(range(len(X_test_patient)), num_samples_to_plot)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        plt.subplot(num_samples_to_plot, 1, i + 1)
        plt.plot(X_test_patient[idx].squeeze(), label="Sygnał EKG")
        plt.title(f"Segment {idx} - Prawdziwe: {y_test_patient[idx]} | Przewidziane: {y_pred[idx]}")
        plt.legend()
    plt.tight_layout()
    plt.show()
