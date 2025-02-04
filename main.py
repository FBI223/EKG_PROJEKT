import numpy as np
import matplotlib.pyplot as plt
import wfdb
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import NUM_SAMPLES
from data.dataset_loader import prepare_qrs_dataset, interpolate_segment
from models.cnn_model import build_cnn
from tensorflow.python.client import device_lib
import tensorflow as tf
import os

from utils.visualization import plot_ecg_record


def main():
    """Główny skrypt uruchamiający przetwarzanie EKG i trening modelu CNN."""

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

    print("Urządzenie domyślne dla obliczeń:", tf.test.gpu_device_name())
    print(device_lib.list_local_devices())


    # Przykłady użycia:
    # 1. Podziel 'MLII' na 60 oddzielnych wykresów (każdy w osobnym oknie)
    #plot_ecg_record('data/raw/mitdb/100', selected_channel='MLII')










    # **1️⃣ Wczytanie danych**
    print("📥 Wczytywanie i segmentacja EKG...")
    X, y = prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES)

    # **2️⃣ Sprawdzenie liczby próbek w każdej klasie**
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))

    print("📊 Liczba próbek w każdej klasie przed filtracją:")
    for cls, count in class_counts_dict.items():
        print(f"Klasa {cls}: {count} próbek")

    # **3️⃣ Usuwanie klas z mniej niż 2 próbkami**
    valid_classes = [cls for cls, count in class_counts_dict.items() if count >= 2]

    # Tworzenie nowej mapy etykiet dla pozostałych klas
    label_map = {cls: i for i, cls in enumerate(valid_classes)}

    # Filtracja danych
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    print(f"✅ Po usunięciu rzadkich klas: {len(X)} segmentów, {len(valid_classes)} klas.")

    # **4️⃣ Konwersja etykiet na indeksy (bez one-hot encoding)**
    y = np.array([label_map[cls] for cls in y])
    num_classes = len(valid_classes)

    # **5️⃣ Podział na zbiór treningowy i testowy**
    print("🔄 Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **6️⃣ Formatowanie danych dla CNN**
    X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, 250, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    # **7️⃣ Definiowanie modelu CNN**
    input_shape = (NUM_SAMPLES, 1)  # 300 próbek na segment, 1 kanał
    model = build_cnn(input_shape, num_classes)

    # **8️⃣ Definicja callbacków (EarlyStopping + ReduceLROnPlateau)**
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # **9️⃣ Trenowanie modelu**
    print("🚀 Rozpoczęcie treningu...")
    history = model.fit(
        X_train, to_categorical(y_train, num_classes),
        epochs=20,
        batch_size=32,
        validation_data=(X_test, to_categorical(y_test, num_classes)),
        callbacks=[early_stopping, reduce_lr]  # ⬅️ Dodane callbacki
    )

    # **🔟 Ewaluacja modelu**
    print("📊 Ewaluacja modelu na zbiorze testowym...")
    loss, accuracy = model.evaluate(X_test, to_categorical(y_test, num_classes))
    print(f"🎯 Test Accuracy: {accuracy:.4f}")








    # **💾 Zapisanie modelu w różnych formatach**
    model.save("models/cnn_ekg.h5")  # HDF5 format
    model.save("models/cnn_ekg.keras")  # Nowy format Keras
    model.save("models/cnn_ekg")  # Format TensorFlow (model jako folder)

    print("✅ Model zapisany w trzech formatach: .h5, .keras, .tf")

    # **📈 Wizualizacja wyników**
    plt.figure(figsize=(10, 5))

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.title("Dokładność CNN")

    # Wykres funkcji straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()
    plt.title("Strata CNN")

    plt.show()


if __name__ == "__main__":
    main()
