import os
import numpy as np
from src.config import PRECOMPUTED_DATA_PATH
from src.logger import setup_logger

def get_filename_suffix(filename):
    return "_".join(filename.replace(".npz", "").split("_")[-2:])

def get_paired_dataset(patient_id: str):
    """
    Loads paired (tf_tensor, bispec_tensor, label) for a specific patient.

    Args:
        patient_id: e.g., "patient_01"

    Returns:
        X_tf: array of time-frequency tensors
        X_bis: array of bispectrum tensors
        y: array of labels (0 = interictal, 1 = preictal)
    """
    logger = setup_logger(name="class_label")
    # Make the Directories
    tf_dir = os.path.join(PRECOMPUTED_DATA_PATH, patient_id, "time-frequency")
    bispec_dir = os.path.join(PRECOMPUTED_DATA_PATH, patient_id, "bispectrum")

    tf_files = [f for f in os.listdir(tf_dir) if f.endswith(".npz")]
    bispec_files = [f for f in os.listdir(bispec_dir) if f.endswith(".npz")]

    tf_map = {get_filename_suffix(f): os.path.join(tf_dir, f) for f in tf_files}
    bis_map = {get_filename_suffix(f): os.path.join(bispec_dir, f) for f in bispec_files}

    common_keys = sorted(set(tf_map.keys()) & set(bis_map.keys()))

    X_tf, X_bis, y = [], [], []

    for key in common_keys:
        tf_path = tf_map[key]
        bis_path = bis_map[key]

        tf_tensor = np.load(tf_path)["tensor"]
        bis_tensor = np.load(bis_path)["tensor"]

        label = 1 if "preictal" in os.path.basename(tf_path) else 0
        #Log for Verification (SORRY IF MADAMI LMAO)
        logger.info(f"Labelled {label} for {os.path.basename(tf_path)} and {os.path.basename(bis_path)}")
        X_tf.append(tf_tensor)
        X_bis.append(bis_tensor)
        y.append(label)

    return np.array(X_tf), np.array(X_bis), np.array(y)

def main():
    patient_id = "patient_01"
    X_tf, X_bis, y = get_paired_dataset(patient_id)

    #Confirm Both Samples are of Equal Count
    print("\nTotal Time-Frequency Samples:", len(X_tf))
    print("Total Bispectrum Samples:", len(X_bis))
    #Check Label Counts
    print("0 - Interictal | 1 - Preictal")
    print("Labels count:", np.unique(y, return_counts=True))

if __name__ == "__main__":
    main()