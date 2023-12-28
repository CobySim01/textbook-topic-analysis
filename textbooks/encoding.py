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


def encode_single_file(source_file, output_file):
    """Encode a single XML file as JSON"""
    print(f"Encoding {source_file}")
    with open(source_file, encoding="utf-8") as f:
        source_xml = BeautifulSoup(f, features="xml")
    parsed_file = parse_xml(source_xml)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_file, f, cls=SoupJSONEncoder, indent=2)
    print(f"Encoded {output_file}")


def encode_files(source_to_output_file: dict[Path, Path]):
    """Encode multiple XML files as JSON using parallel processing."""
    with ProcessPoolExecutor() as executor:
        # Using a dictionary to map futures to their corresponding file details
        future_to_file = {
            executor.submit(encode_single_file, source_file, output_file): source_file
            for source_file, output_file in source_to_output_file.items()
        }

        for future in as_completed(future_to_file):
            old_file = future_to_file[future]
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"File processing failed for {old_file}:\n{exc}")
                raise exc


if __name__ == "__main__":
    TEXTBOOKS_DIRECTORY = Path("textbooks-data")
    PARSED_TEXTBOOKS_DIRECTORY = Path("textbooks-parsed")
    if not PARSED_TEXTBOOKS_DIRECTORY.exists():
        PARSED_TEXTBOOKS_DIRECTORY.mkdir()

    file_mapping = {
        path: PARSED_TEXTBOOKS_DIRECTORY / (path.stem + ".json")
        for path in TEXTBOOKS_DIRECTORY.glob("*.xml")
    }
    encode_files(file_mapping)
