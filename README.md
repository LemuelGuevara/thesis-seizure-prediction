# Multimodal Analysis of EEG via Bispectral Features and Time-frequency Features for Seizure Prediction using EfficientNetB0+CBAM and Gated Fusion

## Environment Setup

Install UV first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment using UV:

```bash
uv sync
```

## Dataset

The EEG recordings obtained are from the publicly available and reliable dataset, CHB-MIT Scalp EEG database, obtained from Physionet.org, a research dataset provider for complex Physiologic Signals.

Dataset link: https://physionet.org/content/chbmit/1.0.0/

## Setting up the config

**Configuration**: Edit the config.toml file to change the preprocessing parameters such as sample rate, paths, and other preprocessing related variables.

> [!NOTE]
> Make sure to place the config.toml in the root directory (outside of the src directory).

Sample config.toml

```toml
[data]
dataset_path = "./dataset/chb-mit-scalp-eeg-database-1.0.0" # path to the raw dataset
precomputed_data_path = "./precomputed_data" # Path to store/load precomputed data
patients_to_process = [1] # indices of patients to process
runs_dir = "./runs"

[preprocessing]
sample_rate = 256 # Sampling rate of EEG signals (Hz)
preictal_minutes = 30 # Duration of preictal period to consider (minutes)
epoch_length = 30 # Duration of each epoch window (seconds)
low_cutoff_filter = 0.5 # Low cutoff frequency for bandpass filter (Hz)
high_cutoff_filter = 32.0 # High cutoff frequency for bandpass filter (Hz)
notch_filter = 60.0 # Frequency for notch filter to remove powerline noise (Hz)
selected_channels = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ"
] # List of EEG channels to use
normalization_method = "minmax" # Method to normalize data (e.g., minmax, zscore)
normalize_power = true

[preprocessing.band_defs]
theta = [4.0, 8.0] # Theta band frequency range (Hz)
alpha = [8.0, 12.0] # Alpha band frequency range (Hz)
beta = [12.0, 30.0] # Beta band frequency range (Hz)

[logging]
logging_level = "INFO" # Logging level (DEBUG, INFO, WARNING, ERROR)

# Values here are not to be changed
[model.classification.basic_conv]
in_planes = 2
out_planes = 1
kernel_size = 7
stride = 1
padding = 0
dilation = 1
groups = 1
relu = true
batch_normalization = true
bias = false
eps = 1e-5
momentum = 0.01
affine = true

# Values here are not to be changed
[model.classification.cbam]
gate_channels = 1280
reduction_ratio = 16
no_spatial = false

# Values here are not to be changed
[model.classification.multi_seizure_model]
feature_dim = 1280
num_classes = 2

# Adjust according the specs of your machine
[model.data.data_loader]
batch_size = 32
shuffle = true
num_workers = 10

# Main training config to be changed accordingly
[model.train]
num_epochs = 50
lr = 1e-4
use_cbam = true
gated = true
modalities = ["tf", "bis"] # Modalities to be used
setup_name = "Setup Name" # Name of the setup
```

> [!NOTE]
> When changing the preprocessing and training params, only edit the config.toml file.
> Also when adding a new config variable in the config.toml file, make sure to also add in the the config.py file.

Flow of preprocessing: precompute stfts -> time-frequency -> bispectrum

> [!NOTE]
> Both time-frequency and bispectrum pipelines will the use the precomputed stfts, so make sure
> that you have run them beforehand.

## Running the Preprocessing Pipelines

The main flow of the entire system pipeline is to first perform the prerocessing that will precompute the stfts, then create the
precomputed time-frequency spectrograms and create the bispectras.

### Precomputed Data Structure

```tree
precomputed_data
├── patient_01 # patient id
│   ├── bispectrum # bispectra .npz
│   ├── stfts # precomputed .npz stfts
│   └── time-frequency # time-frequency spectrograms .npz
```

### Running It

Using bash:

```bash
bash preprocessing_pipeline.sh
```

Using zsh:

```zsh
sh preprocessing_pipeline.sh
```

Once completed, a csv file containing the summarize created preictal and interictal epochs will be created
under the directory `/runs/` with the file `precoputed_stfts.csv`

Sample Table:

| patient_id | number_of_seizures | preictal_epochs | interictal_epochs |
| ---------- | ------------------ | --------------- | ----------------- |
| 01         | 7                  | 647             | 647               |

### Checking the Spectrograms and Bispectra

Open the `check_image.py` file under `src/prerocessing` directory. Then change the following lines according
to your choice:

```python
    [line 122]: patient = (id of the chb case)
    [line 124]: tf_file = f"precomputed_data/patient_{patient_id}/time-frequency/[tf filename of either preictal or interictal].npz"
    [line 125]: bis_file = f"precomputed_data/patient_{patient_id}/bispectrum/[bis filename of either preictal or interictal].npz"
```

## Training

> [!NOTE]
> Before training, make sure to have the precomputed of the patient you want to train on is complete.

### Setting up the Training Config

Change the following variables according to your setup.

```toml
[model.data.data_loader]
batch_size = 32
shuffle = true
num_workers = 10

[model.train]
num_epochs = 50
lr = 1e-4
use_lr_scheduler = false
use_cbam = true
undersample = false
gated = true
class_weighting = false
modalities = ["tf", "bis"]
setup_name = "Setup Name"
```

### Running the Training Pipeline

```bash
uv run -m src.model.train
```

### Checking the Results

To check the results, navigate to the `runs/training` then select the csv file that matches your setup_name e.g. `proposed_setup.csv`

| patient_id | setup_name            | run_timestamp  | true_positives | false_positives | true_negatives | false_negatives | training_accuracy | training_recall | training_f1 | accuracy | recall | f1_score |
| ---------- | --------------------- | -------------- | -------------- | --------------- | -------------- | --------------- | ----------------- | --------------- | ----------- | -------- | ------ | -------- |
| 01         | proposed_setup_proper | Nov25_12-44-15 | 592            | 30              | 597            | 35              | 0.9845            | 0.9845          | 0.9842      | 0.9482   | 0.9442 | 0.948    |

You can also view the graphs and confusion matrix under the directory `runs/training/setup_name/patient_id/`

Sample Structure:

```tree
runs/training/proposed_setup_proper/patient_01
├── patient_01_Nov25_12-44-15_accuracy_graph.png
├── patient_01_Nov25_12-44-15_cf_matrix.png
└── patient_01_Nov25_12-44-15_loss_graph.png
```

## Running the Dashboard GUI

The dashboard GUI is built with gradio and its main purpose is to do inference testing
by using the best model for each patient and using the test cases (either preictal or interictal)
of each patient as the inputfor the model.

> [!NOTE]
> Make sure to still have your precomputed data of time-frequency and bispectrum.

Running the dashboard:

```bash
uv run -m src.dashboard
```

Once the dashboard is running, open the the link in the terminal in your browser of choice. Once
in the browser, the gui will be then shown and there will be 3 file inputs that is required. The first
input needed is the model.pt, this file can be located in `/runs/saved_models/patient_id.pt`; this is the best model
of the patient. Once the has been fed, the next inputs are then the time-frequency spectograms and bispectras. The files
that will be used as inputs are the test cases, so if the test cases of the patient are 10, then the first 10 preictals or interictals
of each modality will be used as the inputs.

> [!NOTE]
> You can only input once phase at a time, so the inputs of each modalities should match phases, no mixing of phases.
> The filename of each epoch should also be the same e.g. `preictal_chb14_18_000075_000105_tf.npz` and `preictal_chb14_18_000075_000105_bis`,
> same file structure, the only difference is the modality which is indicated at the end of the filename e.g. tf or bis.

Once all inputs are fed, you can now do prediction by click the predict button in the dashboard. And the inference confidence will be shown below.
