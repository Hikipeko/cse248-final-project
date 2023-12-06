import time
import re
import json
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from typing import Dict, Tuple
from run_with_parameter import run_and_store_output
from optimizers import TPEOptimizer


def convert_numpy_arrays_to_lists(dictionary):
    """Convert NumPy arrays to Python lists recursively in a dictionary."""
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()
        elif isinstance(value, dict):
            dictionary[key] = convert_numpy_arrays_to_lists(value)
    return dictionary


def write_dict_to_file(data_dict, file_path):
    try:
        # Convert NumPy arrays to lists in the dictionary
        data_dict = convert_numpy_arrays_to_lists(data_dict)

        # Write the modified dictionary to the file with indentation
        with open(file_path, 'w') as file:
            json.dump(data_dict, file)
        print(f"Dictionary written to {file_path} successfully.")
    except Exception as e:
        print(f"Error writing dictionary to {file_path}: {str(e)}")


def read_dict_from_file(file_path):
    try:
        # Read the dictionary from the JSON file
        with open(file_path, 'r') as file:
            data_dict = json.load(file)

        # Convert lists back to NumPy arrays in the dictionary
        data_dict = convert_lists_to_numpy_arrays(data_dict)

        print(f"Dictionary read from {file_path} successfully.")
        return data_dict
    except Exception as e:
        print(f"Error reading dictionary from {file_path}: {str(e)}")
        return None


def convert_lists_to_numpy_arrays(dictionary):
    """Convert Python lists to NumPy arrays recursively in a dictionary."""
    for key, value in dictionary.items():
        if isinstance(value, list):
            dictionary[key] = np.array(value)
        elif isinstance(value, dict):
            dictionary[key] = convert_lists_to_numpy_arrays(value)
    return dictionary


def extract_name_from_path(file_path):
    pattern = r'/([^/]+)\.json$'
    match = re.search(pattern, file_path)
    if match:
        result = match.group(1)
        return result
    else:
        return "temp_log"


def run_wrapper(input_file):
    def wrapper(eval_config):
        return run_and_store_output(eval_config, input_file)
    return wrapper


INPUT_FILES = ['/DREAMPlace/DREAMPlace/test/ispd2005/adaptec1.json',
               '/DREAMPlace/DREAMPlace/test/ispd2005/adaptec2.json',
               '/DREAMPlace/DREAMPlace/test/ispd2005/adaptec3.json',
               '/DREAMPlace/DREAMPlace/test/ispd2005/adaptec4.json']


if __name__ == "__main__":
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"density_weight", lower=-6.0, upper=-1.0))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"gamma", lower=1.0, upper=5.0))

    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"learning_rate", lower=-4.0, upper=-2.0))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"learning_rate_decay", lower=.996, upper=1.0))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"RePlAce_LOWER_PCOF", lower=.9, upper=.99))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
        f"RePlAce_UPPER_PCOF", lower=1.01, upper=1.15))

    metadata = read_dict_from_file('metadata.txt')

    for i in range(4):
        input_file = INPUT_FILES[i % 4]
        task_name = extract_name_from_path(input_file)
        print(
            f"{task_name}=======================================================================")

        opt = TPEOptimizer(
            obj_func=run_wrapper(input_file),
            config_space=cs,
            n_init=10,
            max_evals=100,
            min_bandwidth_factor=1e-2,
            # metadata=metadata,
            resultfile=task_name
        )
        opt.optimize(logger_name=task_name)

        print(opt.fetch_observations())
        metadata[task_name] = opt.fetch_observations()
    write_dict_to_file(metadata, "tpe-adaptec1-4.txt")
