import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import MITDB_PATH
from data.dataset_loader import load_ecg_record, segment_ecg_by_qrs
from data.qrs_detection import detect_qrs_biosppy
from utils.preprocessing import standarize_signal


def test_model_on_patient(model, patient_record):
    """Testuje model na rzeczywistych danych pacjenta z MIT-BIH."""

    # **1️⃣ Załaduj dane pacjenta**
    signal, annotations, labels, fs = load_ecg_record(patient_record, MITDB_PATH)
    signal = standarize_signal(signal[:, 0])
    qrs_peaks = detect_qrs_biosppy(signal, fs)

    # **2️⃣ Segmentacja EKG**
    X_test_patient, y_test_patient = segment_ecg_by_qrs(signal, annotations, labels, qrs_peaks)

    # **3️⃣ Filtrowanie klas, których model nie obsługuje**
    num_classes = model.output_shape[1]
    y_test_patient = np.array([y if y < num_classes else 0 for y in y_test_patient])

    # **4️⃣ Konwersja do one-hot encoding**
    y_test_patient_onehot = to_categorical(y_test_patient, num_classes=num_classes)

    # **5️⃣ Predykcja modelu**
    X_test_patient = np.expand_dims(X_test_patient, axis=-1)
    y_pred = model.predict(X_test_patient, batch_size=16)

    # **6️⃣ Konwersja predykcji do klas**
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_patient_classes = np.argmax(y_test_patient_onehot, axis=1)

    # **7️⃣ Debugowanie**
    print(f"✅ y_test_patient shape: {y_test_patient.shape}, y_pred shape: {y_pred.shape}")
    print(f"✅ Unikalne klasy w y_test_patient_classes: {np.unique(y_test_patient_classes)}")

    # **8️⃣ Wyświetlenie raportu klasyfikacji**
    target_names = [f"Klasa {i}" for i in range(num_classes)]
    print(classification_report(y_test_patient_classes, y_pred_classes, labels=list(range(num_classes)), target_names=target_names, zero_division=0))

    # **9️⃣ Wizualizacja macierzy błędów**
    #plot_confusion_matrix(y_test_patient_classes, y_pred_classes, target_names)


def plot_confusion_matrix(y_true, y_pred, labels):
    """Rysuje macierz błędów."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predykowane klasy")
    plt.ylabel("Prawdziwe klasy")
    plt.title("Macierz błędów")
    plt.show()
