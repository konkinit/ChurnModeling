import os
import sys
import pytest
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data import check_train_frac

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


@pytest.mark.parametrize(
    "frac_input, is_valid",
    [("80", True), ("50.77", False), ("404", False)])
def test_check_train_frac(frac_input, is_valid):
    assert check_train_frac(frac_input) == is_valid

