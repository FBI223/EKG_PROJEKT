import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from data.dataset_loader import prepare_qrs_dataset
from config import NUM_SAMPLES, TESTED_MODEL_PATH, MITDB_PATH
from models.cnn_model import build_cnn
from utils.testing import test_model_on_patient
from tensorflow.keras.models import load_model

def train_model():
    """Trenuje model CNN na EKG."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("✅ Urządzenie domyślne dla obliczeń:", tf.test.gpu_device_name())
    print(device_lib.list_local_devices())
    print("GPU dostępne:", tf.config.list_physical_devices('GPU'))

    # **1️⃣ Wczytanie danych**
    print("📥 Wczytywanie i segmentacja EKG...")
    X, Y = prepare_qrs_dataset(directory=MITDB_PATH)

    # **2️⃣ Mapowanie klas PRZED oversamplingiem**
    unique_classes = np.unique(Y)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    print(f"📌 Mapa klas: {class_mapping}")

    Y = np.array([class_mapping[y] for y in Y])

    # **3️⃣ Oversampling (wyrównanie liczności klas)**
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X, Y = ros.fit_resample(X, Y)

    # **4️⃣ Sprawdzenie liczby próbek po oversamplingu**
    class_counts = Counter(Y)
    print("📊 Liczba próbek po oversamplingu:", class_counts)

    # **5️⃣ Podział na zbiór treningowy i testowy**
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **6️⃣ Zamiana etykiet na one-hot encoding**
    num_classes = len(class_mapping)
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
        X_train, y_train, epochs=6, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping]
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    dataset_dir = MITDB_PATH
    patients = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.dat')]

    for patient in patients:
        test_model_on_patient(model, patient_record=patient)


if __name__ == "__main__":
    #train_model()
    test_model_on_mitdb()
