import pickle
import random
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from config import NUM_SAMPLES, MITDB_PATH
from data.dataset_loader import load_ecg_record, segment_ecg_by_qrs
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal


def test_model_on_patient(model, patient_record):
    """Testuje model na rzeczywistych danych pacjenta z MIT-BIH."""

    # **Załaduj dane pacjenta**
    signal, annotations, labels, fs = load_ecg_record(patient_record, MITDB_PATH)
    signal = standarize_signal(signal[:, 0])
    qrs_peaks = detect_qrs_biosppy(signal, fs)

    # **Segmentacja EKG**
    X_test_patient, y_test_patient = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks)

    num_classes = model.output_shape[1]

    # **Sprawdzenie unikalnych klas w y_test_patient**
    print(f"📊 Unikalne klasy w y_test_patient: {np.unique(y_test_patient)}")
    print(f"📊 Model obsługuje klasy: {num_classes}")

    # **Usuwanie etykiet spoza zakresu modelu**
    y_test_patient = np.array([y if y < num_classes else 0 for y in y_test_patient])

    # **Konwersja do one-hot encoding**
    y_test_patient = to_categorical(y_test_patient, num_classes=num_classes)

    # **Predykcja**
    X_test_patient = np.expand_dims(X_test_patient, axis=-1)
    y_pred = model.predict(X_test_patient, batch_size=16)
    y_pred = (y_pred > 0.5).astype(int)

    # **Debugowanie**
    print(f"✅ y_test_patient shape: {y_test_patient.shape}, y_pred shape: {y_pred.shape}")

    # **Raport**
    target_names = [f"Klasa {i}" for i in range(y_pred.shape[1])]
    print(classification_report(y_test_patient, y_pred, target_names=target_names, zero_division=0))
