import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from collections import Counter

from data.dataset_loader import prepare_qrs_dataset
from config import NUM_SAMPLES, TESTED_MODEL_PATH, MITDB_PATH, SVDB_PATH, LABEL_MAP
from models.cnn_model import build_cnn

# 🔹 **Definicja klas, które chcemy trenować**
SELECTED_CLASSES = list(LABEL_MAP.values())  # Pobranie wartości liczbowych zamiast stringów

def train_model():
    """Trenuje model CNN na wybranych klasach EKG."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("✅ GPU dostępne:", tf.config.list_physical_devices('GPU'))

    # **1️⃣ Wczytanie danych MITDB + SVDB**
    print("📥 Wczytywanie i segmentacja EKG...")
    X_mitdb, Y_mitdb = prepare_qrs_dataset(directory=MITDB_PATH)
    X_svdb, Y_svdb = prepare_qrs_dataset(directory=SVDB_PATH)

    # **2️⃣ Połączenie MITDB + SVDB**
    X = np.concatenate([X_mitdb, X_svdb], axis=0)
    Y = np.concatenate([Y_mitdb, Y_svdb], axis=0)

    # **🔍 DEBUG: Przed filtrowaniem**
    unique_classes = set(Y)
    print(f"📌 Unikalne klasy w danych przed filtrowaniem: {unique_classes}")
    print(f"📊 Liczność klas przed filtrowaniem: {Counter(Y)}")

    # **3️⃣ Filtracja wybranych klas**
    mask = np.isin(Y, SELECTED_CLASSES)  # Użycie `np.isin()` poprawia działanie filtra
    X = X[mask]
    Y = Y[mask]  # Nie trzeba konwertować, bo Y już zawiera liczby

    # **🔍 DEBUG: Po filtrowaniu**
    if len(X) == 0:
        print("❌ Brak danych po filtracji! Sprawdź poprawność `SELECTED_CLASSES` i `LABEL_MAP`.")
        return

    print(f"📌 Finalna liczba próbek po filtracji: {X.shape[0]}")
    print(f"📌 Wybrane klasy (numeryczne): {SELECTED_CLASSES}")
    print(f"📊 Liczność klas po filtrowaniu: {Counter(Y)}")

    # **5️⃣ Podział na zbiór treningowy i testowy**
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **6️⃣ Zamiana etykiet na one-hot encoding**
    num_classes = len(SELECTED_CLASSES)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # **7️⃣ Formatowanie danych dla CNN**
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # **8️⃣ Definiowanie modelu CNN**
    input_shape = (NUM_SAMPLES, 1)
    model = build_cnn(input_shape, num_classes)

    # **9️⃣ CALLBACKI**
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # **🔟 TRENING MODELU**
    print("🚀 Rozpoczęcie treningu...")
    history = model.fit(
        X_train, y_train, epochs=20, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping]
    )

    # **💾 Zapisanie modelu**
    model.save(TESTED_MODEL_PATH + "cnn_ekg_filtered.h5")
    print("✅ Model zapisany!")

    # **📈 Wizualizacja wyników**
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.title("Dokładność CNN")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()
    plt.title("Strata CNN")

    plt.show()

if __name__ == "__main__":
    train_model()
