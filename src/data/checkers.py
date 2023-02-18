def train_frac_check(input: str) -> bool:
    if input.isdigit():
        return 0 < int(input) < 100
    return False
