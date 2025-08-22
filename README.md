# Multimodal Analysis of EEG via Bispectral Features and Time-frequency Features for Seizure Prediction using EfficientNetB0+CBAM and Gated Fusion

## Environment Setup

```zsh
conda env create -f environment.yaml
conda activate seizure-prediction
```

## Configuration

TOML will be used to handle all the config variables. Change values accordingly.

```toml
[data]
dataset_path = "path_of_the_dataset"
specs_output_path = "path_of_spectrograms_output"

[dataset]
number_of_patients = 0 # Change accordingly

[preprocessing]
low_cutoff_filter = 0.0 # Low cutoff frequency for bandpass filter (Hz)
high_cutoff_filter = 0.0 # High cutoff frequency for bandpass filter (Hz)
notch_filter = 0.0 # Frequncy for notch filter to handle powerline noise (Hz)
sample_rate = 0 # EEG sampling frequency (Hz)
preictal_minutes = 0 # Duration of preictal window seizure onset (minutes)
epoch_window_duration_seconds = # Window length for splitting data into epochs (seconds)
selected_channels = [] # List of EEG channels to include
```

## Getting the data

By default, a sample dataset will be in the repository that would consist of a single patient's EEG recordings.

## Running the Code

### Preprocessing

```zsh
python -m src.preprocessing.eeg_to_spectrogram
```

**Configuration**: Edit the config.toml file to change the preprocessing parameters such as sample rate, paths, and other preprocessing related variables.

> [!NOTE]
> When changing the preprocessing params, only edit the config.toml file.

- Outputs: spectrograms that are in the form multiple matrices that contain the stft (NPZ format).

#### Checking How the Spectrogram Looks Like

```zsh
python -m src.preprocessing.read_specs
```
