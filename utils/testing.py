import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from config import NUM_SAMPLES, MITDB_PATH
from data.dataset_loader import load_ecg_record, segment_ecg_by_qrs
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal


def test_model_on_patient(model, patient_record, mlb, known_classes, directory=MITDB_PATH):
    """Testuje model na rzeczywistych danych pacjenta z MIT-BIH."""

    # **Załaduj dane pacjenta**
    signal, annotations, labels, fs = load_ecg_record(patient_record, directory)
    signal = standarize_signal(signal[:, 0])
    qrs_peaks = detect_qrs_biosppy(signal, fs)

    # **Segmentacja EKG**
    X_test_patient, y_test_patient = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks, NUM_SAMPLES)

    # **📌 Konwersja etykiet – usuwamy nieznane klasy**
    filtered_labels = [list(set(y) & known_classes) for y in y_test_patient]
    y_test_patient = mlb.transform(filtered_labels)

    # **Sprawdzenie zgodności wymiarów**
    if y_test_patient.shape[1] != model.output_shape[1]:
        print(f"⚠️ Nieprawidłowa liczba klas w testowym zbiorze! y_test: {y_test_patient.shape}, model expects: {model.output_shape[1]}")
        new_y_test = np.zeros((y_test_patient.shape[0], model.output_shape[1]))
        min_dim = min(y_test_patient.shape[1], model.output_shape[1])
        new_y_test[:, :min_dim] = y_test_patient[:, :min_dim]  # Dopasowanie wymiarów
        y_test_patient = new_y_test

    # **Predykcja**
    X_test_patient = np.expand_dims(X_test_patient, axis=-1)
    y_pred = model.predict(X_test_patient, batch_size=16)
    y_pred = (y_pred > 0.5).astype(int)

    # **Debugowanie**
    print(f"✅ y_test_patient shape: {y_test_patient.shape}, y_pred shape: {y_pred.shape}")

    # **Raport**
    target_names = [f"Klasa {i}" for i in range(y_pred.shape[1])]
    print(classification_report(y_test_patient, y_pred, target_names=target_names, zero_division=0))
