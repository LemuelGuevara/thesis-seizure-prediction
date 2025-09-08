import matplotlib.pyplot as plt
import numpy as np


def view_single_mosaic(npz_path: str, save_png: bool = False) -> None:
    """
    View a single mosaic from npz file.

    Args:
        npz_path: Path to the npz file
        save_png: Whether to save as PNG file
    """
    # Load the npz file
    data = np.load(npz_path)
    tensor = data["tensor"]
    start = data["start"]
    end = data["end"]
    phase = data["phase"]

    print(f"Mosaic shape: {tensor.shape}")
    print(f"Start: {start}, End: {end}, Phase: {phase}")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(tensor[:, :, 0], cmap="viridis")
    plt.title(f"Epoch {start}-{end} (Phase: {phase})")
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)", rotation=270, labelpad=20)
    plt.axis("off")

    if save_png:
        png_path = npz_path.replace(".npz", ".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved PNG: {png_path}")

    plt.show()


if __name__ == "__main__":
    time_frequency = (
        "precomputed_data/patient_01/time-frequency/preictal_tf_band_2951_2981.npz"
    )
    bispctrum = (
        "precomputed_data/patient_01/bispectrum/preictal_bispectrum_2966_2996.npz"
    )

    # Change accordingly
    view_single_mosaic(time_frequency)
