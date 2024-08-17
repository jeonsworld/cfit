from os import PathLike
from typing import Union, Literal, Dict, Any, Optional

from huggingface_hub import get_safetensors_metadata
from huggingface_hub.errors import NotASafetensorsRepoError

from .utils import (
    get_model_file_size,
    get_model_config,
    estimate_parameters,
    calculate_memory,
    convert_bytes_to_human_readable,
    parse_model_size,
    format_model_size,
    Precision,
    PRECISION_MAP,
)


def get_model_size(model_name_or_path: Union[str, PathLike]) -> int:
    """ Get the size of a model file from a Hugging Face repository.
    :param model_name_or_path: The name or path of the model on Hugging Face
    :return: The size of the model file in bytes
    """
    try:
        metadata = get_safetensors_metadata(model_name_or_path)
        return int(metadata.metadata["total_size"])
    except NotASafetensorsRepoError:
        size = get_model_file_size(model_name_or_path)
        if size is None:
            raise ValueError(f"Model size can't be obtained from the model repo: {model_name_or_path}")
        return size
    except Exception as e:
        raise ValueError(f"Error obtaining model size: {str(e)}")


def determine_precision(model_config: Dict[str, Any]) -> int:
    """ Determine the precision of a model based on its configuration.
    :param model_config: The configuration dictionary of the model
    :return: The precision of the model in bits
    """
    if "quantization" in model_config:
        return model_config["quantization"].get("bits", model_config.get("torch_dtype", Precision.BITS_32))
    elif "quantization_config" in model_config:
        if model_config["quantization_config"].get("load_in_4bit"):
            return Precision.BITS_4
        elif model_config["quantization_config"].get("load_in_8bit"):
            return Precision.BITS_8
    return PRECISION_MAP.get(model_config.get("torch_dtype"), Precision.BITS_32)


def from_hf(
    model_name_or_path: Union[str, PathLike],
    precision: Literal["auto", "all", Precision.BITS_32, Precision.BITS_16, Precision.BITS_8, Precision.BITS_4] = "auto",
) -> str:
    """ Calculate the required GPU memory for a model from Hugging Face.
     :param model_name_or_path: The name or path of the model on Hugging Face
     :param precision: The precision to use for calculations ("auto", 32, 16, 8, or 4)
     :return: A string describing the required GPU memory
     """
    model_size = get_model_size(model_name_or_path)

    if precision == "auto":
        model_config = get_model_config(model_name_or_path)
        precision = determine_precision(model_config)

    if precision == "all":
        return calculate_all_precisions(model_size, model_name_or_path)
    else:
        num_params = estimate_parameters(model_size, precision)
        gpu_mem = calculate_memory(num_params, precision)
        return f"Required GPU Memory[{model_name_or_path}, precision: {precision}]: {convert_bytes_to_human_readable(gpu_mem)}"


def from_params(
    num_params: Union[str, int],
    precision: Literal["all", Precision.BITS_32, Precision.BITS_16, Precision.BITS_8, Precision.BITS_4] = "all",
) -> str:
    """ Calculate the required GPU memory based on the number of parameters.
    :param num_params: Number of parameters (as string or int)
    :param precision: The precision to use for calculations ("all", 32, 16, 8, or 4)
    :return: A string describing the required GPU memory
    """
    if isinstance(num_params, str):
        num_params = parse_model_size(num_params)

    if precision == "all":
        return calculate_all_precisions(num_params)
    else:
        gpu_mem = calculate_memory(num_params, precision)
        return f"Required GPU Memory[parameters: {format_model_size(num_params)}, precision: {precision}]: {convert_bytes_to_human_readable(gpu_mem)}"


def calculate_all_precisions(num_params: int, model_name_or_path: Optional[Union[str, PathLike]] = None) -> str:
    """ Calculate the required GPU memory for all precisions.
    :param num_params: Number of parameters
    :param model_name_or_path: The name or path of the model on Hugging Face
    :return: A string describing the required GPU memory for all precisions
    """
    results = []
    for precision in Precision:
        gpu_mem = calculate_memory(num_params, precision)
        results.append(f"  - {precision}bit: {convert_bytes_to_human_readable(gpu_mem)}")
    if model_name_or_path:
        return f"Required GPU Memory[{model_name_or_path}, parameters: {format_model_size(num_params)}]\n" + "\n".join(results)
    else:
        return f"Required GPU Memory[parameters: {format_model_size(num_params)}]\n" + "\n".join(results)
