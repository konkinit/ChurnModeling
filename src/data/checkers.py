def check_train_frac(input: str) -> bool:
    if input.isdigit():
        return 0 < int(input) < 100
    return False
