import json
from typing import Iterable


def write_results(results: Iterable, filename: str):
    """Write results to the given file."""
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
