# Multimodal Analysis of EEG via Bispectral Features and Time-frequency Features for Seizure Prediction using EfficientNetB0+CBAM and Gated Fusion

## Environment Setup

Create the environment using Conda or Mamba:

```bash
# Conda
conda env create -f environment.yaml
conda activate seizure-prediction

# Mamba
mamba env create -f environment.yaml
mamba activate seizure-prediction
```

To update the environment (e.g., after modifying environment.yaml):

```bash
# Using conda
conda env update -f environment.yaml --prune

# OR using mamba
mamba env update -f environment.yaml --prune
```

## Dataset

Link of the EEG recordings: [EEG Recordings](https://drive.google.com/drive/folders/11brbsKB-k09mx-Bo4cyUK2dyVe9vvqe-?usp=drive_link)

> [!NOTE]
> Download the 'dataset' folder in the link.
> Once downloaded and extracted, place it in the root directory (outside of the src folder)

## Running the Code

### Preprocessing

**Configuration**: Edit the config.toml file to change the preprocessing parameters such as sample rate, paths, and other preprocessing related variables.

> [!NOTE]
> When changing the preprocessing params, only edit the config.toml file.
> Also when adding a new config variable in the config.toml file, make sure to also add in the the config.py file.

Flow of preprocessing: precompute stfts -> time-frequency -> bispectrum

> [!NOTE]
> Both time-frequency and bispectrum pipelines will the use the precomputed stfts, so make sure
> that you have run them beforehand.

#### Output Directory Structure
```
precomputed_data/
└── patient_01
    ├── bispectrum
    │   ├── bispectrum_mosaic_epoch_0_30.npz
    ├── stfts
    │   ├── C3-P3
    │   ├── epoch_0_30.npz
    ├── time_frequency
    │   ├── time_frequency_band_epoch_0_30.npz
```

#### Precompute STFTs

Responsible for precomputing stfts beforehand.

```bash
python -m src.preprocessing.run_precompute_stfts_pipeline
```

#### Time-frequency

Converts the stft values from the precomputed stfts into mosaics.

```bash
python -m src.preprocessing.run_time_frequency_pipeline
```

**Output**: .npz containing mosaic of the epoch along with their epoch metadata (start, end, phase) 

#### Bispectrum

Convertes the complex stft coefficients from the precomputed stfts into mosaics. 

**Output**: .npz containing mosaic of the epoch along with their epoch metadata (start, end, phase) 


```bash
python -m src.preprocessing.run_bispectrum_pipeline
```

**Output**: .npz containing mosaic of the epoch along with their epoch metadata (start, end, phase) 


#### Checking How the Mosaic Looks Like

```bash
python -m src.preprocessing.check_mosaic
```