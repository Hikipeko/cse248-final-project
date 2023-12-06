import subprocess
from typing import Dict, Tuple
import json


def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def write_dict_to_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_and_store_output(eval_config: Dict[str, float], input_file):
    script_path = '/DREAMPlace/DREAMPlace/dreamplace/Placer.py'
    output_file = input_file + '.copy'
    result_file = '/DREAMPlace/results/output.txt'

    # Replace 'output.txt' with the desired output file name
    parameters = load_json_from_file(input_file)

    # Process input parameters
    if "density_weight" in eval_config:
        density_weight = eval_config["density_weight"]
        parameters["density_weight"] = 10 ** density_weight

    if "gamma" in eval_config:
        gamma = eval_config["gamma"]
        parameters["gamma"] = gamma

    if "learning_rate" in eval_config:
        learning_rate = eval_config["learning_rate"]
        parameters["global_place_stages"][0]["learning_rate"] = 10 ** learning_rate

    if "learning_rate_decay" in eval_config:
        learning_rate_decay = eval_config["learning_rate_decay"]
        parameters["global_place_stages"][0]["learning_rate_decay"] = learning_rate_decay

    if "RePlAce_LOWER_PCOF" in eval_config:
        RePlAce_LOWER_PCOF = eval_config["RePlAce_LOWER_PCOF"]
        parameters["RePlAce_LOWER_PCOF"] = RePlAce_LOWER_PCOF

    if "RePlAce_UPPER_PCOF" in eval_config:
        RePlAce_UPPER_PCOF = eval_config["RePlAce_UPPER_PCOF"]
        parameters["RePlAce_UPPER_PCOF"] = RePlAce_UPPER_PCOF

    write_dict_to_json_file(parameters, output_file)

    try:
        # Run the Python script and capture the output
        result = subprocess.run(['/opt/conda/bin/python', script_path,
                                output_file], stdout=subprocess.PIPE, text=True, check=True)

        # Store the output to the specified file
        # with open(result_file, 'w') as file:
        #     file.write(result.stdout)

        print(f"Output stored in {output_file}")

        # Split the output into lines and return the last line

        output_lines = result.stdout.splitlines()
        if output_lines:
            final_line = output_lines[-1]
            # print(parameters)
            print(final_line)
            return {"loss": float(final_line)}
        else:
            return {"loss": float('inf')}
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return {"loss": float('inf')}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"loss": float('inf')}


if __name__ == "__main__":
    d = {}
    run_and_store_output(d)
