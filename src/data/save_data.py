import os
import sys
import yaml
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


def save_input_data(key, value) -> None:
    with open(r'./inputs/app_inputs/sample_input.yaml') as file:
        data = yaml.load(file, Loader=yaml.Loader)
    if data is None:
        data = {key: value}
    else:
        data[key] = value
    with open(r'./inputs/app_inputs/sample_input.yaml', 'w') as file:
        yaml.dump(data, file)
