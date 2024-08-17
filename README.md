# cfit(Carbon Fit) - GPU Memory Estimator

**cfit** is a Python tool designed to estimate the GPU memory requirements of machine learning models. By analyzing model files and configurations from Hugging Face repositories, CFIT helps developers determine the memory demands for different precisions (e.g., 32-bit, 16-bit, 8-bit, 4-bit) and model sizes.

## Features

- **Estimate GPU Memory:** Compute the required GPU memory for models hosted on Hugging Face using their file size and configuration.
- **Support for Multiple Precisions:** cfit supports calculating memory usage for 32-bit, 16-bit, 8-bit, and 4-bit precision models.
- **Flexible Input:** Calculate memory requirements based on either the number of parameters or the model file size.
- **Human-Readable Output:** Results are returned in human-readable formats such as GB or MB for easy understanding.


## Installation
The easiest way to install this package is through `pip`, using
```bash
pip install cfit
```
If you want to be sure you're getting the newest version, you can install it directly from github with
```bash
pip install git+https://github.com/jeonsworld/cfit.git
```

## Usage
cfit provides two main ways to estimate the GPU memory requirements:

### CLI
Usage:
```bash
cfit model_or_params -p [precision]
```
Args:
- `model_or_params`: The name of the model on Hugging Face Model Hub or the number of parameters.
- `precision`: The precision of the model (e.g., "all", 32, 16, 8, 4). Default is `all`.

Examples: 
```bash
cfit from_hf HuggingFaceH4/zephyr-7b-beta -p auto
# Required GPU Memory[HuggingFaceH4/zephyr-7b-beta, precision: 16]: 16.19 GB

cfit from_params 175B -p auto
"""
Required GPU Memory[parameters: 175.0B]
  - 32bit: 782.31 GB
  - 16bit: 391.16 GB
  - 8bit: 195.58 GB
  - 4bit: 97.79 GB
"""

cfit from_params 175000000000 -p auto
"""
Required GPU Memory[parameters: 175.0B]
  - 32bit: 782.31 GB
  - 16bit: 391.16 GB
  - 8bit: 195.58 GB
  - 4bit: 97.79 GB
```

### Library
- cfit.from_hf(model_name_or_path, precision)
  - Args:
    - `model_name_or_path`: The name of the model on Hugging Face Model Hub.
    - `precision`: (Optional) The precision of the model (e.g., "auto", "all", 32, 16, 8, 4). Default is `auto`.

Examples:
- `auto` precision
```python3
import cfit

result = cfit.from_hf("HuggingFaceH4/zephyr-7b-beta", precision="auto")  # `auto` parameter estimate precision from model config
print(result)
# Required GPU Memory[HuggingFaceH4/zephyr-7b-beta, precision: 16]: 16.19 GB
```

- `all` precision
```python3
import cfit

result = cfit.from_hf("HuggingFaceH4/zephyr-7b-beta", precision="all")  # `all` parameter estimate memory for all precisions
print(result)
"""
Required GPU Memory[HuggingFaceH4/zephyr-7b-beta, parameters: 14.5B]
  - 32bit: 64.75 GB
  - 16bit: 32.37 GB
  - 8bit: 16.19 GB
  - 4bit: 8.09 GB
"""
```

- Specific precision
```python3
import cfit

result = cfit.from_hf("HuggingFaceH4/zephyr-7b-beta", precision=8)  # estimate memory for specific precision
print(result)
# Required GPU Memory[HuggingFaceH4/zephyr-7b-beta, precision: 8]: 16.19 GB
```

2. From number of parameters
- cfit.from_params(num_params, precision)
  - Args:
    - `num_params`: The number of parameters in the model.
    - `precision`: (Optional) The precision of the model (e.g., "all", 32, 16, 8, 4). Default is `all`.

Examples:
```python3
import cfit

result = cfit.from_params("175B", precision="auto")
"""
Required GPU Memory[parameters: 175.0B]
  - 32bit: 782.31 GB
  - 16bit: 391.16 GB
  - 8bit: 195.58 GB
  - 4bit: 97.79 GB
"""

result = cfit.from_params(175000000000, precision="auto")
print(result)
"""
Required GPU Memory[parameters: 175.0B]
  - 32bit: 782.31 GB
  - 16bit: 391.16 GB
  - 8bit: 195.58 GB
  - 4bit: 97.79 GB
"""
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.