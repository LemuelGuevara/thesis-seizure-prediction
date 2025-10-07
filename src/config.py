"""
Derived config values from config.toml
"""

import os
from dataclasses import dataclass
from typing import ClassVar, Literal

from src.utils import load_toml_config, snake_case

config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
config = load_toml_config(config_path)


@dataclass
class DataConfig:
    dataset_path: ClassVar[str] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", config["data"]["dataset_path"])
    )
    precomputed_data_path: ClassVar[str] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", config["data"]["precomputed_data_path"]
        )
    )
    patients_to_process: ClassVar[list[int]] = config["data"]["patients_to_process"]
    runs_dir: ClassVar[str] = config["data"]["runs_dir"]


@dataclass
class PreprocessingConfig:
    sample_rate: ClassVar[int] = config["preprocessing"]["sample_rate"]
    preictal_minutes: ClassVar[int] = config["preprocessing"]["preictal_minutes"]
    epoch_length: ClassVar[int] = config["preprocessing"]["epoch_length"]
    low_cutoff_filter: ClassVar[float] = config["preprocessing"]["low_cutoff_filter"]
    high_cutoff_filter: ClassVar[float] = config["preprocessing"]["high_cutoff_filter"]
    notch_filter: ClassVar[float] = config["preprocessing"]["notch_filter"]

    # now these are also class-level constants
    selected_channels: ClassVar[list[str]] = config["preprocessing"][
        "selected_channels"
    ]
    normalization_method: ClassVar[Literal["minmax", "zscore"]] = config[
        "preprocessing"
    ]["normalization_method"]
    band_level_bispectrum: ClassVar[bool] = config["preprocessing"][
        "band_level_bispectrum"
    ]
    band_defs: ClassVar[dict[str, list[float]]] = config["preprocessing"]["band_defs"]
    thread_max_workers: ClassVar[int] = os.cpu_count() or 1


@dataclass
class LoggingConfig:
    logging_level: ClassVar[Literal["DEBUG", "VERBOSE", "INFO", "WARNING", "ERROR"]] = (
        config["logging"]["logging_level"]
    )


basic_conv_toml = config["model"]["classification"]["basic_conv"]
cbam_toml = config["model"]["classification"]["cbam"]
multi_seizure_model_toml = config["model"]["classification"]["multi_seizure_model"]
data_loader_toml = config["model"]["data"]["data_loader"]
train_toml = config["model"]["train"]


@dataclass
class BasicConvConfig:
    in_planes: ClassVar[int] = basic_conv_toml["in_planes"]
    out_planes: ClassVar[int] = basic_conv_toml["out_planes"]
    kernel_size: ClassVar[int] = basic_conv_toml["kernel_size"]
    stride: ClassVar[int] = basic_conv_toml["stride"]
    padding: ClassVar[int] = basic_conv_toml["padding"]
    dilation: ClassVar[int] = basic_conv_toml["dilation"]
    groups: ClassVar[int] = basic_conv_toml["groups"]
    relu: ClassVar[bool] = basic_conv_toml["relu"]
    batch_normalization: ClassVar[bool] = basic_conv_toml["batch_normalization"]
    bias: ClassVar[bool] = basic_conv_toml["bias"]
    eps: ClassVar[float] = basic_conv_toml["eps"]
    momentum: ClassVar[float] = basic_conv_toml["momentum"]
    affine: ClassVar[bool] = basic_conv_toml["affine"]


@dataclass
class CBAMConfig:
    gate_channels: ClassVar[int] = cbam_toml["gate_channels"]
    reduction_ratio: ClassVar[int] = cbam_toml["reduction_ratio"]
    no_spatial: ClassVar[bool] = cbam_toml["no_spatial"]


@dataclass
class MultiSeizureModelConfig:
    feature_dim: ClassVar[int] = multi_seizure_model_toml["feature_dim"]
    num_clsses: ClassVar[int] = multi_seizure_model_toml["num_classes"]


@dataclass
class DataLoaderConfig:
    batch_size: ClassVar[int] = data_loader_toml["batch_size"]
    shuffle: ClassVar[bool] = data_loader_toml["shuffle"]
    num_workers: ClassVar[bool] = data_loader_toml["num_workers"]


@dataclass
class Trainconfig:
    num_epochs: ClassVar[int] = train_toml["num_epochs"]
    lr: ClassVar[float] = train_toml["lr"]
    use_cbam: ClassVar[bool] = train_toml["use_cbam"]
    undersample: ClassVar[bool] = train_toml["undersample"]
    gated: ClassVar[bool] = train_toml["gated"]
    class_weighting: ClassVar[bool] = train_toml["class_weighting"]
    setup_name: ClassVar[str] = snake_case(train_toml["setup_name"])
