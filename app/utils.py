def parse_key_value_pairs(raw_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for chunk in raw_text.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def as_float(values: dict[str, str], key: str, default: float | None = None) -> float:
    if key not in values:
        if default is None:
            raise ValueError(f"Missing required field: {key}")
        return default
    return float(values[key])


def as_list(values: dict[str, str], key: str) -> list[str]:
    if key not in values or not values[key]:
        return []
    return [item.strip() for item in values[key].replace(",", ";").split(";") if item.strip()]


def as_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    if key not in values:
        return default
    return values[key].strip().lower() in {"1", "true", "yes", "on"}
