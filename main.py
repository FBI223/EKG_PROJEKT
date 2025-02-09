import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.client import device_lib
from data.dataset_loader import prepare_qrs_dataset
from config import NUM_SAMPLES, MIN_SAMPLES_FOR_MODEL
from models.cnn_model import build_cnn_multilabel, f1_score
from utils.preprocessing import filter_rare_classes
from utils.testing import test_model_on_patient
from tensorflow.keras.models import load_model





def train_model():
    """Trenuje model CNN na EKG."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ✅ Ukrycie ostrzeżeń TensorFlow

    print("✅ Urządzenie domyślne dla obliczeń:", tf.test.gpu_device_name())
    print(device_lib.list_local_devices())
    print("GPU dostępne:", tf.config.list_physical_devices('GPU'))
    print()

    # **1️⃣ Wczytanie danych**
    print("📥 Wczytywanie i segmentacja EKG...")
    X, Y = prepare_qrs_dataset(directory="data/raw/mitdb/", num_samples=NUM_SAMPLES)

    # **2️⃣ Usuwanie rzadkich klas**
    X, Y = filter_rare_classes(X, Y, min_samples=MIN_SAMPLES_FOR_MODEL)

    # **3️⃣ Podział na zbiór treningowy i testowy**
    print("🔄 Podział na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"📦 Zbiór treningowy: {X_train.shape[0]} próbek, Zbiór testowy: {X_test.shape[0]} próbek")

    # **4️⃣ Formatowanie danych dla CNN**
    X_train = np.expand_dims(X_train, axis=-1)  # (num_samples, 300, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    # **5️⃣ Definiowanie modelu CNN**
    input_shape = (NUM_SAMPLES, 1)  # 300 próbek na segment, 1 kanał
    model = build_cnn_multilabel(input_shape, Y.shape[1])

    # **6️⃣ CALLBACKI**
    early_stopping = EarlyStopping(
        monitor='val_auc', patience=5, restore_best_weights=True, verbose=1, mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # **7️⃣ TRENING MODELU**
    print("🚀 Rozpoczęcie treningu...")
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
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




def test_model_on_mitdb():
    """Wczytuje zapisany model i testuje go na pacjentach."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ Wyłącza GPU dla TensorFlow w testowaniu

    # 📂 Ścieżka do modelu
    model_path_h5 = "models/trained_models/v7/cnn_ekg.h5"
    model_path_keras = "models/trained_models/v7/cnn_ekg.keras"

    # 🔍 Sprawdzenie, który model istnieje
    if os.path.exists(model_path_h5):
        model = load_model(model_path_h5, custom_objects={"f1_score": f1_score})
        print("✅ Wczytano model z:", model_path_h5)
    elif os.path.exists(model_path_keras):
        model = load_model(model_path_keras, custom_objects={"f1_score": f1_score})
        print("✅ Wczytano model z:", model_path_keras)
    else:
        raise FileNotFoundError("❌ Nie znaleziono modelu w v7!")

    # 📥 Wczytaj `mlb.pkl` raz, na początku
    with open("models/mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    known_classes = set(mlb.classes_)


    # 🔍 Pobranie listy wszystkich pacjentów w bazie
    dataset_dir = "data/raw/mitdb/"
    patients = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.dat')]

    # **💉 Testowanie na wszystkich pacjentach**
    for patient in patients:
        test_model_on_patient(model, patient_record=patient, mlb=mlb, known_classes=known_classes)




"""


ID Klasy	Symbol MIT-BIH	Opis klasyfikacji
0	N	Normalny rytm zatokowy
1	L	Lewa odnogi bloku pęczka Hisa (LBBB)
2	R	Prawa odnoga bloku pęczka Hisa (RBBB)
3	A	Przedwczesne pobudzenie przedsionkowe
4	V	Przedwczesne pobudzenie komorowe
5	/	Aberracja (nieprawidłowa morfologia QRS)
6	f	Przedwczesne pobudzenie nadkomorowe
7	e	Ektopowa arytmia
8	j	Idiowentrykularny rytm
9	E	Przedwczesne pobudzenie komorowe
10	S	Skurcz dodatkowy
11	Q	Nieokreślony
12	F	Migotanie przedsionków
13	!	Blok przedsionkowo-komorowy
14		
15	'x'	Nieznane zaburzenie
16	'P'	Pacemaker Spike
17	'T'	Transmisja zakłócona
18	'+'	Rytm AV Dissociation
19	'U'	Nietypowe zaburzenie
20	'~'	Szum lub niestabilność
21	'`'	Inne zdarzenie
22	'?'	Niezidentyfikowane

"""

if __name__ == "__main__":
    train_model()  # 🚀 Trening modelu
    test_model_on_mitdb()   # 🔬 Testowanie modelu
