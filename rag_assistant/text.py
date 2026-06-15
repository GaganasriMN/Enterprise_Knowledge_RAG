import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_%-]+", text.lower())
