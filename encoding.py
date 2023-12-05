import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from parse import parse_xml


class SoupJSONEncoder(json.JSONEncoder):
    """Encode bs4.element.Tag as string when serialising JSON."""

    def default(self, o):
        if isinstance(o, Tag):
            return str(o)
        return super().default(o)


def process_single_file(file_pair):
    source_file, output_file = file_pair
    with open(source_file, encoding="utf-8") as f:
        source_xml = BeautifulSoup(f, features="xml")
    parsed_file = parse_xml(source_xml)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_file, f, cls=SoupJSONEncoder, indent=2)


def process_files(file_mapping: dict[Path, Path]):
    with ProcessPoolExecutor() as executor:
        # Using a dictionary to map futures to their corresponding file details
        future_to_file = {
            executor.submit(process_single_file, file_pair): file_pair[0]
            for file_pair in file_mapping.items()
        }

        for future in as_completed(future_to_file):
            old_file = future_to_file[future]
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"File processing failed for {old_file}:\n{exc}")
                raise exc
