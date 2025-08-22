import os

import matplotlib.pyplot as plt
import numpy as np

base = os.path.dirname(__file__)
path = os.path.join(
    base, "..", "..", "specs", "patient_01", "preictal", "spec_pr_0_30.npz"
)
data = np.load(os.path.abspath(path), allow_pickle=True)

# Example: if your npz has frequency, time, and magnitude
f = data["freqs"]  # frequency bins
t = data["times"]  # time bins
Sxx = data["stft"]  # spectrogram in dB

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, Sxx, shading="gouraud")
plt.title(f"Channel: {data['channel']}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
