import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib
from data.dataset_loader import prepare_qrs_dataset
from config import NUM_SAMPLES, MIN_SAMPLES_FOR_MODEL, TESTED_MODEL_PATH, MITDB_PATH
from models.cnn_model import build_cnn
from utils.preprocessing import filter_rare_classes
from utils.testing import test_model_on_patient
from tensorflow.keras.models import load_model

def train_model():
    """Trenuje model CNN na EKG."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ✅ Ukrycie ostrzeżeń TensorFlow

    print("✅ Urządzenie domyślne dla obliczeń:", tf.test.gpu_device_name())
    print(device_lib.list_local_devices())
    print("GPU dostępne:", tf.config.list_physical_devices('GPU'))

    # **1️⃣ Wczytanie danych**
    print("📥 Wczytywanie i segmentacja EKG...")
    X, Y = prepare_qrs_dataset(directory=MITDB_PATH)


    # **2️⃣ Usuwanie rzadkich klas**
    X, Y = filter_rare_classes(X, Y, min_samples=MIN_SAMPLES_FOR_MODEL)

    # **🔍 Debug: Sprawdzenie unikalnych klas po filtracji**
    unique_classes = np.unique(Y)
    num_classes = len(unique_classes)
    print(f"🎯 Unikalne klasy po filtracji: {unique_classes}, liczba klas: {num_classes}")

    # ✅ Mapowanie klas na indeksy 0, 1, 2, ... N-1
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    print(f"📌 Mapa klas: {class_mapping}")

    # Zamiana wartości Y na indeksy od 0 do num_classes-1
    Y = np.array([class_mapping[y] for y in Y])

    # **3️⃣ Podział na zbiór treningowy i testowy**
    print("🔄 Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **🎯 Zamiana etykiet na one-hot encoding**
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # **4️⃣ Formatowanie danych dla CNN**
    X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, 300, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    # **5️⃣ Definiowanie modelu CNN**
    input_shape = (NUM_SAMPLES, 1)  # 300 próbek na segment, 1 kanał
    model = build_cnn(input_shape, num_classes)

    # **6️⃣ CALLBACKI**
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # **7️⃣ TRENING MODELU**
    print("🚀 Rozpoczęcie treningu...")
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

    # **💾 Zapisanie modelu**
    model.save(TESTED_MODEL_PATH + "cnn_ekg.h5")
    model.save(TESTED_MODEL_PATH + "cnn_ekg.keras")
    print("✅ Model zapisany w formatach: .h5, .keras")

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

def test_model_on_mitdb():
    """Wczytuje zapisany model i testuje go na pacjentach."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ Wyłącza GPU dla TensorFlow w testowaniu

    # 📂 Ścieżka do modelu
    model_path_h5 = TESTED_MODEL_PATH + "cnn_ekg.h5"
    model_path_keras = TESTED_MODEL_PATH + "cnn_ekg.keras"

    if os.path.exists(model_path_h5):
        model = load_model(model_path_h5)
        print("✅ Wczytano model z:", model_path_h5)
    elif os.path.exists(model_path_keras):
        model = load_model(model_path_keras)
        print("✅ Wczytano model z:", model_path_keras)
    else:
        raise FileNotFoundError("❌ Nie znaleziono modelu!")

    # 🔍 Pobranie listy wszystkich pacjentów w bazie
    dataset_dir = MITDB_PATH
    patients = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.dat')]

    # **💉 Testowanie na wszystkich pacjentach**
    for patient in patients:
        test_model_on_patient(model, patient_record=patient)

if __name__ == "__main__":
    #train_model()  # 🚀 Trening modelu
    test_model_on_mitdb()
