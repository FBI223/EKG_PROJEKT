import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np


def describe_annotations(annotations):
    """
    Dodaje opisy do znalezionych adnotacji w MIT-BIH.

    :param annotations: Słownik {adnotacja: liczba wystąpień}
    """
    descriptions = {
        'N': "Normalny rytm serca (QRS)",
        'V': "Przedwczesne pobudzenie komorowe (PVC)",
        'S': "Nadkomorowe pobudzenie",
        'F': "Fusion beat (hybrydowe pobudzenie)",
        'Q': "Nieokreślona arytmia",
        '~': "Artefakt (zakłócenia sygnału)",
        '|': "Separator rytmu",
        '/': "Początek nowego segmentu rytmu",
        'f': "Flutter (przelotna arytmia)",
        '+': "Komentarz lekarza (ignorowany)",
        'A': "Pobudzenie nadkomorowe",
        'E': "Pobudzenie ektopowe",
        'J': "Junctional beat (z węzła AV)",
        'L': "Lewogram (wariant QRS)",
        'R': "Pobudzenie rytmu węzłowego",
        'P': "Pobudzenie przedsionkowe",
        'B': "Blok przedsionkowo-komorowy",
        'T': "T-wave (zmiany w załamku T)",
        'Z': "Nieznana zmiana",
    }

    print("\n📌 **Lista znalezionych adnotacji:**")
    for label, count in annotations.items():
        description = descriptions.get(label, "Brak opisu (może być niestandardowe)")
        print(f" - **{label}** ({count} razy): {description}")


def plot_ecg_with_labels(record_name, signal, annotations, labels, fs=360, start_time=None, end_time=None, padding=2):
    """
    Rysuje wykres EKG z buforem i poprawionym układem wykresu.

    :param record_name: Nazwa rekordu
    :param signal: Sygnał EKG (pojedynczy kanał)
    :param annotations: Lista pozycji próbek z adnotacjami
    :param labels: Lista etykiet anomalii
    :param fs: Częstotliwość próbkowania (np. 360 Hz)
    :param start_time: Początek wycinka (w sekundach) - domyślnie od początku
    :param end_time: Koniec wycinka (w sekundach) - domyślnie do końca
    :param padding: Dodatkowy bufor po lewej i prawej stronie (w sekundach)
    """

    # 🔹 Jeśli `start_time` lub `end_time` nie są podane, ustawiamy pełny zakres
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = len(signal) / fs  # Cały sygnał

    # 🔹 Konwersja czasu na próbki
    start_sample = max(0, int((start_time - padding) * fs))  # Uwzględniamy padding (ale nie poniżej 0)
    end_sample = min(len(signal), int((end_time + padding) * fs))  # Uwzględniamy padding

    # 🔹 Wycinamy fragment sygnału
    time_axis = np.arange(start_sample, end_sample) / fs
    signal_cut = signal[start_sample:end_sample]

    # 🔹 Tworzenie wykresu z poprawionym paddingiem
    fig, ax = plt.subplots(figsize=(310, 4))
    ax.plot(time_axis, signal_cut, color='black', linewidth=1, label="Sygnał EKG")

    # 🔹 Poprawa marginesów wokół wykresu
    ax.margins(0.05)  # Dodaje przestrzeń wokół wykresu
    plt.subplots_adjust(left=0.07, right=0.80, top=0.9, bottom=0.2)  # Padding

    # 🔹 Mapowanie kolorów dla adnotacji
    label_colors = {
        'N': 'green', 'V': 'purple', 'Q': 'brown', 'A': 'cyan', 'F': 'orange',
        'f': 'blue', 'S': 'lime', 'J': 'gold', 'L': 'pink', 'R': 'darkblue',
        'j': 'lightblue', 'E': 'red', '/': 'magenta', '|': 'yellow',
        '~': 'gray', '!': 'crimson', 'a': 'olive', 'x': 'darkred',
        '[': 'black', ']': 'black', '"': 'teal', 'e': 'darkgreen'
    }

    label_markers = {
        'N': 'o',  # ✅ Normalne pobudzenie QRS (standardowy skurcz serca)
        'V': 'D',  # ⚠️ Przedwczesne pobudzenie komorowe (PVC) – nieprawidłowe skurcze komorowe
        'Q': 'x',  # ❓ Nieokreślona arytmia (może być różnymi rodzajami zaburzeń)
        'A': '^',  # 🔼 Pobudzenie nadkomorowe (SVEB) – nadkomorowe pobudzenie ektopowe
        'F': '*',  # ⭐ Fusion beat (pobudzenie mieszane) – skurcz będący wynikiem dwóch sygnałów
        'f': 'v',  # 🔽 Flutter przedsionkowy – szybkie, ale regularne drgania przedsionków
        'S': 's',  # 🟢 Nadkomorowe pobudzenie ektopowe – inny rodzaj SVEB
        'J': 'h',  # 🔄 Junctional escape beat – pobudzenie z okolicy węzła przedsionkowo-komorowego
        'L': 'p',  # ⬅️ Lewogram – przemieszczenie osi serca w lewo
        'R': 'H',  # ➡️ Prawogram – przemieszczenie osi serca w prawo
        'j': '8',  # 🟠 Junkcyjne pobudzenie – z obszaru węzła AV
        'E': 'P',  # ❌ Ektopowe pobudzenie komorowe – bardzo nieprawidłowe pobudzenie komorowe
        '/': 'X',  # 🔁 Separator nowego segmentu rytmu
        '|': '|',  # 🔀 Separator rytmu – wskazuje początek nowego rytmu serca
        '~': '_',  # 🔊 Artefakt – zakłócenia w sygnale (np. ruch elektrody, zakłócenia elektryczne)
        '!': '+',  # 🚀 Tachykardia – szybkie bicie serca (>100 BPM)
        'a': '<',  # ⏪ Aberrant beat – nietypowe pobudzenie nadkomorowe
        'x': '>',  # ❓ Nieznana anomalia – rzadko spotykane pobudzenie (brak oficjalnej dokumentacji)
        '[': '1',  # 📌 Otworzenie segmentu – początek analizy odcinka sygnału
        ']': '2',  # 📌 Zamknięcie segmentu – koniec analizy odcinka sygnału
        '"': '3',  # 📝 Specyficzne zdarzenie analizy (np. zmiana ustawień detekcji)
        'e': '4'   # 🛑 Escape beat – pobudzenie ratunkowe serca (gdy naturalny rytm zwalnia)
    }


    # 🔹 Rysowanie adnotacji w wybranym zakresie czasu
    used_labels = set()
    for ann, label in zip(annotations, labels):
        ann_time = ann / fs
        if start_time - padding <= ann_time < end_time + padding and label in label_colors:
            color = label_colors[label]
            marker = label_markers[label]

            # Rysowanie oznaczenia
            ax.scatter(ann_time, signal[ann], color=color, marker=marker, s=80, label=label if label not in used_labels else "")
            used_labels.add(label)

    # 🔹 Konfiguracja wykresu
    ax.set_title(f"Sygnał EKG: {record_name} ({start_time}s - {end_time}s)")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")

    # **Nowa szerokość z paddingiem**
    ax.set_xlim(start_time - padding, end_time + padding)

    # 🔹 Poprawa czytelności: legenda poza wykresem
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)  # Przesunięcie legendy poza wykres

    # 🔹 Eksport do pliku SVG
    plt.savefig(f"{record_name}_{start_time}s_{end_time}s_ekg.svg", format='svg', bbox_inches='tight')

    plt.show()





def plot_qrs_cycles(segments):
    """
    Rysuje wykres pierwszych 5 cykli serca po segmentacji QRS.
    """
    plt.figure(figsize=(20, 6))
    for i in range(min(3, len(segments))):
        plt.plot(segments[i], label=f"Cykł {i+1}")

    plt.legend()
    plt.title("Segmentacja na pełne cykle serca (QRS → QRS)")
    plt.xlabel("Próbki w cyklu")
    plt.ylabel("Amplituda")
    plt.show()


