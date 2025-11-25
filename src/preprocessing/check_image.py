import os

import matplotlib.pyplot as plt
import numpy as np


def load_npz_data(npz_path: str):
    """Load tensor and metadata from npz file."""
    data = np.load(npz_path)
    tensor = np.squeeze(data["tensor"])
    start = float(data["start"][()])  # Extract 0-d array
    end = float(data["end"][()])  # Extract 0-d array
    phase = str(data["phase"][()])  # Extract 0-d array
    freqs = data["freqs"]  # This is a 1-d array
    n_channels = data["n_channels"]
    return tensor, start, end, phase, freqs, n_channels


def plot_spectrogram(ax, tensor, start, end, phase, freqs, n_channels):
    """Plot RGB spectrogram (freq × time × 3)."""
    print(f"Spectrogram shape: {tensor.shape}")

    # Check if truly RGB or grayscale duplicated
    if tensor.shape[-1] == 3:
        ch_diff = (
            np.abs(tensor[..., 0] - tensor[..., 1]).sum()
            + np.abs(tensor[..., 1] - tensor[..., 2]).sum()
        )
        is_rgb = ch_diff > 1e-6
        print(
            f"RGB check: {'True RGB' if is_rgb else 'Grayscale duplicated'} (channel diff: {ch_diff:.2e})"
        )

    duration_sec = end - start
    img = ax.imshow(
        tensor[..., 0],
        origin="lower",
        aspect="auto",
        extent=[0, duration_sec, freqs[0], freqs[-1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram: {phase} ({start:.1f}-{end:.1f}s)")
    plt.colorbar(img, ax=ax, label="Power (dB)")


def plot_bispectrum(ax, tensor, phase, start, end, freqs):
    """Plot RGB bispectrum (freq × freq × 3)."""
    print(f"Bispectrum shape: {tensor.shape}")

    # Check if truly RGB or grayscale duplicated
    if tensor.shape[-1] == 3:
        ch_diff = (
            np.abs(tensor[..., 0] - tensor[..., 1]).sum()
            + np.abs(tensor[..., 1] - tensor[..., 2]).sum()
        )
        is_rgb = ch_diff > 1e-6
        print(
            f"RGB check: {'True RGB' if is_rgb else 'Grayscale duplicated'} (channel diff: {ch_diff:.2e})"
        )

    duration_sec = end - start
    freq_max = freqs[-1]
    img = ax.imshow(
        tensor[..., 0],
        cmap="viridis",
        origin="lower",
        aspect="equal",
        extent=[0, freq_max, 0, freq_max],
    )
    ax.set_xlabel("f₁ (Hz)")
    ax.set_ylabel("f₂ (Hz)")
    ax.set_title(f"Bispectrum: {phase} ({start:.1f}-{end:.1f}s)")
    plt.colorbar(img, ax=ax, label="Power (dB)")


def view_timefreq_and_bispectrum(tf_npz: str | None = None, bis_npz: str | None = None):
    """
    View time-frequency spectrogram and bispectrum side by side (if both provided).
    """
    if not tf_npz and not bis_npz:
        raise ValueError("At least one of tf_npz or bis_npz must be provided.")

    n_plots = sum([tf_npz is not None, bis_npz is not None])
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if tf_npz:
        tf_tensor, start, end, phase, freqs, n_channels = load_npz_data(tf_npz)

        # ➜ add this block:
        if tf_tensor.ndim == 3 and tf_tensor.shape[0] in (1, 3):
            tf_tensor = np.moveaxis(tf_tensor, 0, -1)  # (C,H,W) -> (H,W,C)

        if tf_tensor.ndim != 3 or tf_tensor.shape[2] != 3:
            raise ValueError(
                f"Expected RGB spectrogram (H, W, 3), got {tf_tensor.shape}"
            )
        plot_spectrogram(axes[idx], tf_tensor, start, end, phase, freqs, n_channels)
        idx += 1

    if bis_npz:
        bis_tensor, start, end, phase, freqs, n_channels = load_npz_data(bis_npz)

        # ➜ add this block:
        if bis_tensor.ndim == 3 and bis_tensor.shape[0] in (1, 3):
            bis_tensor = np.moveaxis(bis_tensor, 0, -1)  # (C,H,W) -> (H,W,C)

        if bis_tensor.ndim != 3 or bis_tensor.shape[2] != 3:
            raise ValueError(
                f"Expected RGB bispectrum (H, W, 3), got {bis_tensor.shape}"
            )
        plot_bispectrum(axes[idx], bis_tensor, phase, start, end, freqs)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    patient = 14
    patient_id = f"{patient:02d}"
    tf_file = f"precomputed_data/patient_{patient_id}/time-frequency/preictal_chb14_18_000075_000105_tf.npz"
    bis_file = f"precomputed_data/patient_{patient_id}/bispectrum/interictal_chb14_42_002850_002880_bis.npz"

    # view both if available
    if os.path.exists(tf_file) and os.path.exists(bis_file):
        view_timefreq_and_bispectrum(tf_npz=tf_file, bis_npz=bis_file)
    elif os.path.exists(tf_file):
        view_timefreq_and_bispectrum(tf_npz=tf_file)
    elif os.path.exists(bis_file):
        view_timefreq_and_bispectrum(bis_npz=bis_file)
    else:
        print("No valid files found.")
