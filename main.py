from data.dataset_loader import load_ecg_record
from models.cnn_model import build_cnn
from utils.preprocessing import normalize_signal, filter_signal
from utils.visualization import plot_ecg
import numpy as np


def main():
    """Główny skrypt uruchamiający analizę"""
    record_name = '100'  # Nazwa rekordu MIT-BIH
    ecg_signal, annotations, fs = load_ecg_record(record_name)  # Wczytanie sygnału

    # Sprawdzenie liczby kanałów
    num_channels = ecg_signal.shape[1]
    print(f"Liczba kanałów w {record_name}: {num_channels}")

    # Utworzenie nowego kanału jako średniej ze wszystkich dostępnych kanałów
    avg_channel = np.mean(ecg_signal, axis=1)

    # Normalizacja i filtracja nowego kanału
    avg_channel = normalize_signal(avg_channel)
    avg_channel = filter_signal(avg_channel)

    # Rysowanie tylko średniego kanału
    plot_ecg(record_name, avg_channel)

if __name__ == "__main__":
    main()
