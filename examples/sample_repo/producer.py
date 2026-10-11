"""Small structural fixture; not a benchmark or a production integration."""
def normalize_name(value: str) -> str:
    return value.strip().title()
