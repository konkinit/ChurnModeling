import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def tests_file_present():
    """
    test the presence of the data/inputs file
    """
    required_files = []
    with open("required_files.txt", "r") as f:
        required_files.extend(line.strip() for line in f)
    print(required_files)
    for file in required_files:
        assert os.path.isfile(os.path.join(file)), f"{file} is not present"