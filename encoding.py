import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from parse import parse_xml
from textbooks import Section, Textbook


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


def convert_json_to_textbook(json_file_path: Path) -> Textbook:
    """Parses JSON serialized textbook as Textbook object"""
    with open(json_file_path, encoding="utf-8") as file:
        data = json.load(file)
    textbook = Textbook(json_file_path.stem)

    sections_dict = {}
    for section_id, section_data in data.items():
        new_section = Section(
            section_id=section_id,
            header=section_data["header"],
            # content_xml=section_data["content_xml"],
            content_string=section_data["content_string"],
            word_count=section_data["word_count"],
            subsections=section_data["subsections"],
            annotations=section_data["annotations"],
        )
        sections_dict[section_id] = new_section

    textbook.build_hierarchy(sections_dict, data)

    # Add top-level sections to textbook
    for section_id, section in sections_dict.items():
        # Assuming top-level sections are those not listed as a subsection of any other section
        if not any(section_id in s_data["subsections"] for s_data in data.values()):
            textbook.add_section(section)

    textbook.assign_section_numbers(sections_dict)

    return textbook
