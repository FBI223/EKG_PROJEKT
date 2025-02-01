import numpy as np
from scipy.signal import medfilt

def normalize_signal(ecg_signal):
    """Normalizuje sygnał EKG"""
    return (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))

def filter_signal(ecg_signal):
    """Filtruje sygnał EKG usuwając szumy"""
    return medfilt(ecg_signal, kernel_size=5)