# config.py - tutaj przechowujemy stałe globalne
NUM_SAMPLES = 300  # Stała liczba próbek
FS = 360  # Częstotliwość próbkowania MIT-BIH
MIN_SAMPLES_FOR_MODEL=10
MITDB_PATH="data/raw/mitdb/"
ICENTIA11K_PATH="data/raw/mitdb/"
SHAOXING_PATH="data/raw/mitdb/"
TESTED_MODEL_PATH="models/trained_models/v2_no_interpolation/"

LABEL_MAP = {
    'N': 0,    # Normal beat (Normalny rytm zatokowy)
    'L': 1,    # Left bundle branch block beat (LBBB) (Lewa odnoga bloku pęczka Hisa)
    'R': 2,    # Right bundle branch block beat (RBBB) (Prawa odnoga bloku pęczka Hisa)
    'A': 3,    # Atrial premature beat (Przedwczesne pobudzenie przedsionkowe)
    'a': 4,    # Aberrated atrial premature beat (Aberracja pobudzenia przedsionkowego)
    'J': 5,    # Nodal (junctional) premature beat (Przedwczesne pobudzenie węzłowe)
    'S': 6,    # Supraventricular premature beat (Przedwczesne pobudzenie nadkomorowe)
    'V': 7,    # Premature ventricular contraction (Przedwczesne pobudzenie komorowe)
    'F': 8,    # Fusion of ventricular and normal beat (Fuzja pobudzenia komorowego i normalnego)
    '[': 9,    # Start of ventricular flutter/fibrillation (Początek trzepotania/migotania komór)
    '!': 10,   # Ventricular flutter wave (Fala trzepotania komór)
    ']': 11,   # End of ventricular flutter/fibrillation (Koniec trzepotania/migotania komór)
    'e': 12,   # Atrial escape beat (Pobudzenie ucieczkowe przedsionkowe)
    'j': 13,   # Nodal (junctional) escape beat (Pobudzenie ucieczkowe węzłowe)
    'E': 14,   # Ventricular escape beat (Pobudzenie ucieczkowe komorowe)
    '/': 15,   # Paced beat (Pobudzenie stymulatorowe)
    'f': 16,   # Fusion of paced and normal beat (Fuzja pobudzenia stymulatorowego i normalnego)
    'x': 17,   # Non-conducted P-wave (blocked APB) (Blokowany załamek P)
    'Q': 18,   # Unclassifiable beat (Nieokreślony rytm)
    '|': 19,   # Isolated QRS-like artifact (Izolowany artefakt przypominający QRS)
    '+': 20,   # Rytm AV Dissociation (Rozkojarzenie przedsionkowo-komorowe)
    '~': 21,   # Noise (Szum)
    '?': 22    # Unidentified (Niezidentyfikowane zdarzenie)
}

# Definiujemy hierarchię priorytetów
CLASS_PRIORITY = [
    [9, 10, 11, 7, 8],     # Poważne zaburzenia rytmu
    [1, 2],                # Bloki odnóg pęczka Hisa
    [3, 4, 6, 5],          # Pobudzenia nadkomorowe
    [12, 13, 14],          # Pobudzenia ucieczkowe
    [18, 17, 20, 19],      # Artefakty i nieokreślone pobudzenia
    [15, 16],              # Pobudzenia stymulatorowe
    [0],                   # Normalny rytm zatokowy
    [22, 21]               # Szum i niezidentyfikowane zdarzenia
]

