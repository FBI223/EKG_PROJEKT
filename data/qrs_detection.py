import biosppy
import heartpy as hp
import numpy as np
from scipy.signal import find_peaks
import scipy.signal as signal

def detect_qrs_biosppy(ecg_signal, fs):
    """
    Wykrywa zespoły QRS za pomocą BioSPPy.
    """
    out = biosppy.signals.ecg.ecg(ecg_signal, sampling_rate=fs, show=False)
    return out[2]  # Indeksy wykrytych zespołów QRS




def detect_qrs_heartpy(ecg_signal, fs):
    """
    Wykrywa zespoły QRS za pomocą HeartPy.
    """
    wd, _ = hp.process(ecg_signal, fs)
    return wd['peaklist']  # Indeksy QRS




def detect_qrs(signal, fs):
    """
    Wykrywa zespoły QRS w sygnale EKG przy użyciu różnych metod.

    :param signal: Sygnał EKG (numpy array)
    :param fs: Częstotliwość próbkowania (Hz)
    :param method: Metoda wykrywania ("biosppy", "neurokit", "heartpy", "scipy")
    :return: Tablica indeksów próbek z wykrytymi QRS (numpy array)
    """
    rpeaks, _ = find_peaks(signal, height=0.5, distance=fs * 0.4)


    print(f"QRS peaks detected: {rpeaks}")
    print(f"Number of QRS peaks: {len(rpeaks)}")
    for i in range(len(rpeaks) - 1):
        start = rpeaks[i]
        end = rpeaks[i + 1]
        print(f"Segment {i}: start={start}, end={end}, length={end - start}")

    return np.array(rpeaks, dtype=int)  # Zwracamy w tym samym formacie, co find_peaks


def pan_tompkins_qrs(ecg_signal, fs):
    """
    Wykrywa zespoły QRS w sygnale EKG przy użyciu algorytmu Pan-Tompkins.

    :param ecg_signal: Sygnał EKG (numpy array)
    :param fs: Częstotliwość próbkowania (Hz)
    :return: Indeksy wykrytych zespołów QRS
    """

    # 1. **Filtracja pasmowa 5-15 Hz (dla usunięcia szumów)**
    b, a = signal.butter(1, [5 / (fs / 2), 15 / (fs / 2)], btype='bandpass')
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)

    # 2. **Pochodna sygnału (wykrywanie stromych zmian)**
    derivative_ecg = np.diff(filtered_ecg)

    # 3. **Kwadrat sygnału (podkreślenie załamków QRS)**
    squared_ecg = derivative_ecg ** 2

    # 4. **Przesuwna suma całkowa (uśrednienie sygnału)**
    integration_window = int(0.15 * fs)  # 150 ms okno całkowania
    integrated_ecg = np.convolve(squared_ecg, np.ones(integration_window)/integration_window, mode='same')

    # 5. **Wykrywanie pików (próg dynamiczny)**
    threshold = 0.6 * np.max(integrated_ecg)  # Dynamiczny próg
    peaks, _ = signal.find_peaks(integrated_ecg, height=threshold, distance=fs*0.4)  # Min. 400ms odstęp między QRS

    return peaks


