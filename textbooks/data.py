import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DISALLOWED_SECTION_HEADERS = {
    "exercises",
    "solutions",
    "index",
    "glossary",
    "references",
    "appendix",
}


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
        excluded_sections = []
        for section_id, section_data in data.items():
            if any(h in section_data["header"] for h in DISALLOWED_SECTION_HEADERS):
                excluded_sections.append(section_id)
                continue
            new_section = Section(
                section_id=section_id,
                header=section_data["header"],
                content=section_data["content"],
                word_count=section_data["word_count"],
                subsections=section_data["subsections"],
                concepts=section_data["concepts"],
            )
            sections_dict[section_id] = new_section

        for section_id, section in sections_dict.items():
            section.subsections = [
                s for s in section.subsections if s not in excluded_sections
            ]
        for section in excluded_sections:
            data.pop(section)

        textbook.build_hierarchy(sections_dict, data)
        # Add top-level sections to textbook
        for section_id, section in sections_dict.items():
            # Assuming top-level sections are those not listed as a subsection of any other section
            if not any(section_id in s_data["subsections"] for s_data in data.values()):
                textbook.add_section(section)
        textbook.assign_section_numbers(sections_dict)

        return textbook

    def add_section(self, section: "Section"):
        """Adds a section to this textbook's sections"""
        self.subsections.append(section)
        section.textbook = self

    def all_subsections(self) -> list["Section"]:
        """Flattens all sections into a single list."""
        return [
            sub for section in self.subsections for sub in section.all_subsections()
        ]

    def assign_section_numbers(self, sections_dict):
        """Assigns section numbers to top-level sections and their subsections"""
        section_number = 1
        for section in self.subsections:
            if section.section_id in sections_dict:
                sections_dict[section.section_id].assign_section_number(
                    (section_number,)
                )
                section_number += 1

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

    section_id: str = field(compare=False, repr=True)
    header: str = field(compare=False, repr=True)
    content: str = field(compare=False, repr=False)
    word_count: int = field(compare=False, repr=False)
    subsections: list["Section"] = field(compare=False, repr=False)
    concepts: dict[str, dict[str, str]] = field(compare=False, repr=False)
    section_number: Optional[tuple[int, ...]] = field(default=None, repr=True)
    textbook: Optional[Textbook] = field(default=None)

    def all_subsections(self) -> list["Section"]:
        """Returns a list of all subsections."""
        return [self] + [
            sub for sec in self.subsections for sub in sec.all_subsections()
        ]

    def assign_section_number(self, number):
        """Assigns a section number based on position in the textbook hierachy."""
        self.section_number = number
        for i, subsection in enumerate(self.subsections, start=1):
            subsection.assign_section_number(tuple(list(number) + [i]))

    def print_entry(self, indent=""):
        """Prints a textual representation for a section's TOC entry"""
        section_number_string = ".".join(str(s) for s in self.section_number)
        print(f"{indent}{section_number_string}: {self.header}")

    def __hash__(self) -> int:
        return hash((self.textbook, self.section_id))

    def __eq__(self, other) -> bool:
        return (self.textbook, self.section_id) == (other.textbook, other.section_id)
