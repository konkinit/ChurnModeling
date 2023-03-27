import os
import sys
import pytest
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.features import indicator_ab


@pytest.mark.parametrize(
    "x, a, b, flag",
    [(1.5, 2.0, 4.5, 0), (2, 0.7, 11, 1)])
def test_indicator_ab(x, a, b, flag):
    assert indicator_ab(x, a, b) == flag
