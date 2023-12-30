import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from textbooks.utils import (
    remove_section_number,
    section_number_string_to_tuple,
    extract_section_number,
    is_valid_entry,
)


@dataclass
class Textbook:
    """Represents a Textbook document."""

    name: str
    subsections: list["Section"] = field(default_factory=list, repr=False, init=False)

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ValueError("name atribute must be a string")

    @classmethod
    def from_json(cls, path: Path) -> "Textbook":
        """Loads JSON serialized textbook as Textbook object"""
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        textbook = cls(path.stem)

        sections_dict = {}
        for section_id, section_data in data.items():
            new_section = Section(
                section_id=section_id,
                entry=section_data["entry"],
                header=remove_section_number(section_data["entry"]),
                number=section_number_string_to_tuple(
                    extract_section_number(section_data["entry"])
                ),
                level=section_data["level"],
                is_valid=is_valid_entry(section_data["entry"]),
                content=section_data["content"],
                word_count=section_data["word_count"],
                subsections=section_data["subsections"],
                concepts=section_data["concepts"],
            )
            sections_dict[section_id] = new_section

        textbook.build_hierarchy(sections_dict, data)
        # Add top-level sections to textbook
        for section_id, section in sections_dict.items():
            # Assuming top-level sections are those not listed as a subsection of any other section
            if not any(section_id in s_data["subsections"] for s_data in data.values()):
                textbook.add_section(section)

        return textbook

    def add_section(self, section: "Section"):
        """Adds a section to this textbook's sections"""
        self.subsections.append(section)
        section.textbook = self

    def all_subsections(self) -> list["Section"]:
        """Flattens all sections into a single list."""
        return [
            sub
            for section in self.subsections
            for sub in section.all_subsections()
            if sub.is_valid
        ]

    def build_hierarchy(self, sections_dict, data):
        """Use subsection data to populate the subsections attributes"""
        for section_id, section_data in data.items():
            section = sections_dict[section_id]
            section.subsections = [
                sections_dict[sub_id]
                for sub_id in section_data["subsections"]
                if sub_id in sections_dict
            ]
            for s in section.subsections:
                s.textbook = self

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Section:  # pylint: disable=too-many-instance-attributes
    """Represents a section in a textbook."""

    section_id: str = field(compare=False)
    entry: str = field(compare=False)
    header: str = field(compare=False, repr=False)
    number: Optional[tuple[int | str, ...]] = field(repr=False)
    level: int = field(compare=False)
    is_valid: bool = field(compare=False, repr=False)
    content: str = field(compare=False, repr=False)
    word_count: int = field(compare=False, repr=False)
    subsections: list["Section"] = field(compare=False, repr=False)
    concepts: dict[str, dict[str, str]] = field(compare=False, repr=False)
    textbook: Optional[Textbook] = field(default=None)

    def all_subsections(self) -> list["Section"]:
        """Returns a list of all subsections."""
        return [self] + [
            sub for sec in self.subsections for sub in sec.all_subsections()
        ]

    def assign_section_number(self, number):
        """Assigns a section number based on position in the textbook hierachy."""
        self.number = number
        for i, subsection in enumerate(self.subsections, start=1):
            subsection.assign_section_number(tuple(list(number) + [i]))

    def print_entry(self, indent=""):
        """Prints a textual representation for a section's TOC entry"""
        if self.number is not None:
            section_number_string = ".".join(str(s) for s in self.number)
        else:
            section_number_string = "---"
        print(f"{indent}{section_number_string}:", end=" ")
        print("" if self.is_valid else "[excluded]", self.header)

    def __hash__(self) -> int:
        return hash((self.textbook, self.section_id))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Section):
            return False
        return (self.textbook, self.section_id) == (other.textbook, other.section_id)
