import json
from typing import Optional

from textbooks.data import Section, Textbook

BOOK_COLUMNS = ["section_id", "title"]


def find_section(entry: str, textbook: Textbook = None) -> Optional[Section]:
    """Finds a section in Textbook by the full text representation of its header."""
    if textbook is None:
        raise ValueError
    print(textbook.all_subsections())
    for section in textbook.all_subsections():
        if entry == section.entry and section.is_valid:
            return section
    return None


def get_expert_mapping(base_textbook, other_textbook):
    with open("evaluation-data/expert-mapping.json", encoding="utf-8") as f:
        mappings = json.load(f)
    reoriented_mappings = {}
    for mapping in mappings:
        base_section = find_section(mapping["base_title"], base_textbook)
        other_section = find_section(mapping["other_title"], other_textbook)
        if base_section is None or other_section is None:
            continue
        reoriented_mappings.setdefault(base_section, {})[other_section] = mapping[
            "data"
        ]
    return reoriented_mappings
