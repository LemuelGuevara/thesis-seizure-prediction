"""
Runner script that loops through patients and calls the TF and bispectrum branches.
Assumes data_cleaning.py has: load_patient_summary, extract_seizure_intervals, load_patient_recording
and data_transformation.py has precompute_stfts_for_all_patients.
"""
import os
import argparse
from utils import setup_logger
from data_cleaning import load_patient_summary, extract_seizure_intervals, load_patient_recording
from data_transformation import (
    precompute_stfts_for_all_patients,
    SELECTED_CHANNELS,
    SAMPLE_RATE,
    EPOCH_WINDOW_DURATION_SECONDS,
)
from timefrequency_branch import process_patient_timefrequency
from bispectrum_branch import process_patient_bispectrum

logger = setup_logger("run_tf_pipeline", level=logging.INFO)


def build_labeled_epochs_for_patient(patient_id: str, dataset_path: str, epoch_len: int, overlap_frac: float):
    """
    Load patient summary and extract labeled epochs (preictal/interictal) using extract_seizure_intervals.
    Returns the list of epoch dicts and the concatenated recording (for precompute).
    """
    with load_patient_summary(patient_id, dataset_path) as f:
        preictal_intervals, interictal_intervals, ictal_intervals = extract_seizure_intervals(f)

    # build epoch list
    epochs = []
    def segment_itv(itv, phase):
        step = max(1, int(epoch_len * (1 - overlap_frac)))
        cur = int(itv.start)
        while cur + epoch_len <= itv.end:
            epochs.append({"start": cur, "end": cur + epoch_len, "phase": phase})
            cur += step

    for itv in interictal_intervals:
        segment_itv(itv, "interictal")
    for itv in preictal_intervals:
        segment_itv(itv, "preictal")
    epochs.sort(key=lambda x: x["start"])

    # load recording for this patient (used for precompute)
    recording = load_patient_recording(patient_id)
    return epochs, recording


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    precomputed_root = os.path.join(args.output_dir, "precomputed_stft")

    # Optionally precompute STFTs
    if args.precompute:
        logger.info("Precomputing STFTs for all patients...")
        def builder(pid):
            # returns (epochs, recording) tuple as required by precompute helper
            return build_labeled_epochs_for_patient(pid, args.dataset_path, args.epoch_len, args.overlap_frac)
        precompute_stfts_for_all_patients(
            number_of_patients=args.num_patients,
            dataset_path=args.dataset_path,
            out_root=precomputed_root,
            selected_channels=SELECTED_CHANNELS,
            labeled_epochs_builder=builder,
            sample_rate=args.sample_rate,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            compute_power=True,
            normalize_power_flag=True,
        )

    # Loop patients and run both branches
    for idx in range(1, args.num_patients + 1):
        pid = f"{idx:02d}"
        logger.info("Processing patient %s", pid)
        epochs, recording = build_labeled_epochs_for_patient(pid, args.dataset_path, args.epoch_len, args.overlap_frac)

        # time-frequency branch (band mode)
        out_tf = os.path.join(args.output_dir, f"patient_{pid}", "tf")
        os.makedirs(out_tf, exist_ok=True)
        process_patient_timefrequency(
            patient_id=pid,
            labeled_epochs=epochs, 
            out_dir=out_tf,
            mode="band",
            channel_names=SELECTED_CHANNELS,
            precomputed_stft_root=(precomputed_root if args.precompute else None),
            stft_store=None,  # not used when precomputed root is provided
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            convert_to_power=True,
            normalize_power=True,
            per_tile_normalization=False,
            tile_norm_method="minmax",
            global_normalize_mosaic=False,
            grid=(4,4),
            resize_to=(224,224),
            manifest=True,
            manifest_filename="manifest_tf.csv",
            skip_existing=True,
        )

        # bispectrum branch
        out_bis = os.path.join(args.output_dir, f"patient_{pid}", "bispec")
        os.makedirs(out_bis, exist_ok=True)
        process_patient_bispectrum(
            patient_id=pid,
            labeled_epochs=epochs,
            out_dir=out_bis,
            channel_names=SELECTED_CHANNELS,
            precomputed_stft_root=(precomputed_root if args.precompute else None),
            stft_store=None,
            grid=(4,4),
            per_tile_normalization=True,
            tile_norm_method="minmax",
            global_normalize_mosaic=True,
            resize_to=(224,224),
            manifest=True,
            manifest_filename="manifest_bispec.csv",
            skip_existing=True,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )

    logger.info("All patients processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--output_dir", default=os.environ.get("SPECS_OUTPUT_PATH", "./specs"))
    parser.add_argument("--num_patients", type=int, default=24)
    parser.add_argument("--precompute", action="store_true", help="Precompute STFTs for all patients (recommended)")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--nperseg", type=int, default=SAMPLE_RATE)   # e.g. 256
    parser.add_argument("--noverlap", type=int, default=SAMPLE_RATE//2)
    parser.add_argument("--epoch_len", type=int, default=EPOCH_WINDOW_DURATION_SECONDS)
    parser.add_argument("--overlap_frac", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
