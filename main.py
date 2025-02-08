import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.client import device_lib
from data.dataset_loader import prepare_qrs_dataset
from config import NUM_SAMPLES
from models.cnn_model import build_cnn_multilabel
from utils.testing import test_model_on_patient


def main():
    """Główny skrypt uruchamiający przetwarzanie EKG i trening modelu CNN."""

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ukrycie ostrzeżeń TensorFlow

    print("✅ Urządzenie domyślne dla obliczeń:", tf.test.gpu_device_name())
    print(device_lib.list_local_devices())
    print("GPU dostępne:", tf.config.list_physical_devices('GPU'))
    print()

    # **1️⃣ Wczytanie danych**
    print("📥 Wczytywanie i segmentacja EKG...")
    X, Y = prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES)

    print(f"✅ Po `prepare_qrs_dataset()`: X.shape={X.shape}, Y.shape={Y.shape}")

    # **2️⃣ Sprawdzenie liczby próbek w każdej klasie**
    class_counts = np.sum(Y, axis=0)  # Każda kolumna w Y to osobna klasa (multi-label)
    print(f"📊 Liczba próbek w każdej klasie przed filtracją: {class_counts}")

    # **3️⃣ Usuwanie rzadkich klas**
    valid_classes = np.where(class_counts >= 10)[0]  # Wybierz klasy z co najmniej 10 próbkami
    print(f"🎯 Klasy po filtracji: {valid_classes}")

    # **4️⃣ Tworzenie maski dla multi-label**
    mask = np.any(Y[:, valid_classes] == 1, axis=1)  # Segmenty, które mają co najmniej jedną ważną klasę
    print(f"🔍 Mask.shape: {mask.shape}, True values: {np.sum(mask)}")

    # **5️⃣ Filtrowanie**
    X, Y = X[mask], Y[mask]
    print(f"✅ Po filtracji: X.shape={X.shape}, Y.shape={Y.shape}")

    # **6️⃣ Podział na zbiór treningowy i testowy**
    print("🔄 Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )  # 🚀 **Usunięcie `stratify`, bo mamy multi-label**

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **7️⃣ Formatowanie danych dla CNN**
    X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, 300, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    # **8️⃣ Definiowanie modelu CNN**
    input_shape = (NUM_SAMPLES, 1)  # 300 próbek na segment, 1 kanał
    model = build_cnn_multilabel(input_shape, Y.shape[1])

    # **9️⃣ CALLBACKI**
    early_stopping = EarlyStopping(
        monitor='val_auc', patience=5, restore_best_weights=True, verbose=1, mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        filepath="models/trained_models/v6_multilabel/best_model.h5", monitor='val_auc', save_best_only=True, verbose=1, mode='max'
    )

    # **🔟 TRENING MODELU**
    print("🚀 Rozpoczęcie treningu...")
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # **💾 Zapisanie modelu**
    model.save("models/cnn_ekg.h5")
    model.save("models/cnn_ekg.keras")

    print("✅ Model zapisany w formatach: .h5, .keras")

    # **📈 Wizualizacja wyników**
    plt.figure(figsize=(10, 5))

    # Wykres AUC
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.xlabel("Epoki")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("AUC CNN")

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
