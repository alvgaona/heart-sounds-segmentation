import matplotlib.pyplot as plt
import pandas as pd
import ssq
import torch
from scipy import signal


if __name__ == "__main__":
    df = pd.read_csv("./resources/data/springer_sounds/0001.csv", skiprows=1, names=["Signals", "Labels"])
    x = torch.tensor(df.loc[:, "Signals"].to_numpy())
    y = torch.tensor(df.loc[:, "Labels"].to_numpy(), dtype=torch.int64)

    window = signal.get_window(("kaiser", 0.5), 128, fftbins=False)
    s, f, t = ssq.fsst(x.numpy(), 1000, window)

    # Plot input signal
    plt.figure(figsize=(12, 6))
    plt.plot(x)
    plt.title("Input Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, abs(s), shading="gouraud")
    plt.colorbar(label="Magnitude")
    plt.title("Spectrogram")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
