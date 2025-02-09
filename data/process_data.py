
SAVE_PATH = "processed_data/"

import os
import pickle
import numpy as np

def save_processed_data(X, Y, filename):
    """
    Zapisuje przetworzone segmenty EKG i etykiety do pliku.

    :param X: Przetworzone segmenty EKG (numpy array)
    :param Y: Odpowiednie etykiety (numpy array)
    :param filename: Nazwa pliku (bez rozszerzenia)
    """
    os.makedirs("data/processed", exist_ok=True)  # Tworzy folder jeśli nie istnieje
    filepath = f"data/processed/{filename}.npz"

    np.savez_compressed(filepath, X=X, Y=Y)  # Zapis do pliku `.npz`
    print(f"✅ Dane zapisane: {filepath}")


def load_processed_data(filename):
    """
    Wczytuje przetworzone segmenty EKG i etykiety z pliku.

    :param filename: Nazwa pliku (bez rozszerzenia)
    :return: X, Y, mlb (lub None jeśli plik nie istnieje)
    """
    filepath = f"data/processed/{filename}.npz"

    if not os.path.exists(filepath):
        print("⚠️ Brak zapisanych danych. Przetwarzanie od nowa.")
        return None, None, None

    data = np.load(filepath)
    print(f"✅ Wczytano zapisane dane: {filepath}")

    # Wczytanie `MultiLabelBinarizer`
    mlb_path = "models/mlb.pkl"
    if os.path.exists(mlb_path):
        with open(mlb_path, "rb") as f:
            mlb = pickle.load(f)
        print("✅ Wczytano MultiLabelBinarizer")
    else:
        print("⚠️ Brak zapisanych etykiet `mlb.pkl`.")
        mlb = None

    return data["X"], data["Y"], mlb
