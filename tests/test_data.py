import os


def tests_file_present():
    """
    test the presence of the data file
    """
    required_files = []
    with open("required_files.txt", "r") as f:
        required_files.extend(line.strip() for line in f)
    for file in required_files:
        assert os.path.isfile(os.path.join(file)), f"{file} is not present"