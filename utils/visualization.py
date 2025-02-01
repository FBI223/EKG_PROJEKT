import matplotlib.pyplot as plt
import wfdb
import os



def plot_ecg(record_name, avg_channel):
    """
    Rysuje wykres sygnału EKG po uśrednieniu dwóch kanałów.

    :param record_name: Nazwa rekordu (np. '100')
    :param avg_channel: Połączony kanał (średnia dwóch)
    """
    N_OF_SAMPLES = 648_000

    samples = len(avg_channel)

    plt.figure(figsize=(12, 5))
    plt.plot(avg_channel, label="Średni kanał (1+2)", linewidth=2, color="black")

    plt.title(f"Sygnał EKG: {record_name} ({samples} próbek)")
    plt.xlabel("Próbki")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.show()


