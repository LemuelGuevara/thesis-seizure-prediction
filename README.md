# Multimodal Analysis of EEG via Bispectral Features and Time-frequency Features for Seizure Prediction using EfficientNetB0+CBAM and Gated Fusion

## Environment Setup

```zsh
conda env create -f environment.yaml
conda activate seizure-prediction
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
