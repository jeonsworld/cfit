import math
import re
from enum import IntEnum, auto
from os import PathLike
from typing import Optional, Dict, Union

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url


class ModelFileType(IntEnum):
    PYTORCH = auto()
    TENSORFLOW = auto()
    FLAX = auto()
    RUST = auto()
    CHECKPOINT = auto()
    ONNX = auto()
    COREML = auto()


MODEL_FILE_NAMES: Dict[ModelFileType, str] = {
    ModelFileType.PYTORCH: "pytorch_model.bin",
    ModelFileType.TENSORFLOW: "tf_model.h5",
    ModelFileType.FLAX: "flax_model.msgpack",
    ModelFileType.RUST: "rust_model.ot",
    ModelFileType.CHECKPOINT: "model.ckpt",
    ModelFileType.ONNX: "model.onnx",
    ModelFileType.COREML: "coreml_model.mlmodel",
}


class Precision(IntEnum):
    BITS_32 = 32
    BITS_16 = 16
    BITS_8 = 8
    BITS_4 = 4


PRECISION_MAP: Dict[str, Precision] = {
    "float32": Precision.BITS_32,
    "float16": Precision.BITS_16,
    "bfloat16": Precision.BITS_16,
    "int8": Precision.BITS_8,
    "int4": Precision.BITS_4,
}


class ModelSizeUnit(IntEnum):
    MILLION = 10 ** 6
    BILLION = 10 ** 9
    TRILLION = 10 ** 12


def get_model_file_size(model_name_or_path: Union[str, PathLike]) -> Optional[int]:
    """ Get the size of a model file from a Hugging Face repository.
    :param model_name_or_path: The name or path of the model on Hugging Face
    :return: The size of the model file in bytes, or None if not found
    """
    api = HfApi()
    try:
        model_info = api.model_info(model_name_or_path)
        for sibling in model_info.siblings:
            if sibling.rfilename in MODEL_FILE_NAMES.values():
                metadata = get_hf_file_metadata(hf_hub_url(model_name_or_path, sibling.rfilename))
                return metadata.size
    except Exception as e:
        raise ValueError(f"Failed to get model file size: {e}")
    return None


def get_model_config(model_name_or_path: Union[str, PathLike]) -> Dict:
    """ Get the configuration of a model from a Hugging Face repository.
    :param model_name_or_path: The name or path of the model on Hugging Face
    :return: The model configuration as a dictionary
    """
    config_url = hf_hub_url(model_name_or_path, "config.json")
    response = requests.get(config_url)
    if response.status_code == 200:
        return response.json()
    raise ValueError(f"Config file not found in the model repo: {response.text}")


def calculate_memory(total_parameters: int, precision: Precision) -> float:
    """ Calculate the memory required for a model.
    :param total_parameters: The total number of parameters in the model
    :param precision: The precision of the model (in bits)
    :return: The estimated memory requirement in bytes
    """
    return (total_parameters * 4) / (32 / precision) * 1.2


def convert_bytes_to_human_readable(bytes_value: float) -> str:
    """ Convert bytes to a human-readable string (GB or MB).
    :param bytes_value: The number of bytes
    :return: A string representation of the size in GB or MB
    """
    gb = bytes_value / (1024 ** 3)
    mb = bytes_value / (1024 ** 2)
    return f"{gb:.2f} GB" if gb >= 1 else f"{mb:.2f} MB"


def parse_model_size(size_str: str) -> int:
    """ Convert a string representation of model size to an integer.
    :param size_str: A string representing model size (e.g., "10M", "2B", "1.5T")
    :return: The size as an integer
    """
    size_str = size_str.lower()
    match = re.match(r"(\d+(\.\d+)?)([mbt])", size_str)
    if not match:
        raise ValueError(f"Invalid model size format: {size_str}")
    number, unit = float(match.group(1)), match.group(3)
    size_mapping = {"m": ModelSizeUnit.MILLION, "b": ModelSizeUnit.BILLION, "t": ModelSizeUnit.TRILLION}
    return int(number * size_mapping[unit])


def format_model_size(size_int: int) -> str:
    """ Convert an integer representation of model size to a human-readable string.
    :param size_int: An integer representing model size
    :return: A string representation of the size (e.g., "10M", "2B", "1.5T")
    """
    size_mapping = [
        (ModelSizeUnit.TRILLION, "T"),
        (ModelSizeUnit.BILLION, "B"),
        (ModelSizeUnit.MILLION, "M")
    ]
    for threshold, unit in size_mapping:
        if size_int >= threshold:
            return f"{size_int / threshold:.1f}{unit}"
    return str(size_int)


def estimate_parameters(file_size_bytes: int, precision: Precision) -> int:
    """ Estimate the number of parameters in a model based on file size and precision.
    :param file_size_bytes: The size of the model file in bytes
    :param precision: The precision of the model (in bits)
    :return: The estimated number of parameters
    """
    total_bits = file_size_bytes * 8
    estimated_parameters = total_bits / precision
    return math.ceil(estimated_parameters)
