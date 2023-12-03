import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, DefaultDict, Optional

import numpy as np


@dataclass
class Textbook:
    """Represents a Textbook document."""

    name: str
    sections: dict[str, "Section"] = field(default_factory=dict, repr=False)

    def add_section(self, section):
        """Adds a section to this textbooks sections"""
        self.sections[section.section_id] = section
        section.textbook = self

    def print_toc(self):
        """Prints a textual representation of the Textbook's table of contents."""
        for section in self.sections.values():
            section.print_entry()
            section.print_subsections(indent="\t")

    @property
    def flattened_sections(self):
        attributes = []

        def recurse(subsection):
            attributes.append(subsection)
            for subsub in subsection.subsections:
                recurse(subsub)

        for section in self.sections.values():
            recurse(section)
        return attributes

    def __hash__(self) -> str:
        return hash(self.name)


@dataclass(unsafe_hash=True)
class Section:  # pylint: disable=too-many-instance-attributes
    """Represents a section in a textbook."""

    section_id: str = field(compare=False, repr=True)
    header: str = field(compare=False, repr=True)
    content_xml: str = field(compare=False, repr=False)
    content_string: str = field(compare=False, repr=False)
    word_count: int = field(compare=False, repr=False)
    subsections: list["Section"] = field(compare=False, repr=False)
    annotations: list[str] = field(compare=False, repr=False)
    section_number: Optional[tuple[int, ...]] = field(default=None, repr=True)
    textbook: Optional[Textbook] = field(default=None, hash=False)

    def assign_section_number(self, number):
        """Assigns a section number based on position in the textbook hierachy."""
        self.section_number = number
        for i, subsection in enumerate(self.subsections, start=1):
            subsection.assign_section_number(tuple(list(number) + [i]))

    def print_entry(self, indent=""):
        """Prints a textual representation for a section's TOC entry"""
        section_number_string = ".".join(str(s) for s in self.section_number)
        print(f"{indent}{section_number_string}: {self.header}")

    def print_subsections(self, indent=""):
        """Prints the subsections for a textbook"""
        for section in self.subsections:
            section.print_entry(indent)
            section.print_subsections(indent + "\t")


@dataclass
class TextbookWithSectionVectors:
    """Represents a Textbook with section vectors computed using some method."""

    textbook: Textbook
    compute_vector: Callable[[Section], Any]
    section_vectors: dict[Section, Any] = field(
        default_factory=dict, repr=False, init=False
    )

    def __post_init__(self):
        for section in self.textbook.sections.values():
            self._compute_section_vectors(section)

    def _compute_section_vectors(self, section):
        weights = []
        for subsection in section.subsections:
            self._compute_section_vectors(subsection)
            weights.append(subsection.word_count)
        vectors = [
            self.section_vectors[subsection] for subsection in section.subsections
        ]
        this_section_vector = self.compute_vector(section)
        vectors.append(this_section_vector)
        this_section_weight = section.word_count
        weights.append(this_section_weight)
        if sum(weights) > 0:
            aggregated_vector = np.average(vectors, weights=weights)
        else:
            aggregated_vector = 0
        self.section_vectors |= {section: aggregated_vector}


@dataclass
class IntegratedTextbook:
    """Represents a Textbook integrated with sections from other textbooks."""

    textbook: TextbookWithSectionVectors
    similarity_function: Callable[[str, str], float]
    similarity_threshold: float
    section_mapping: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )

    def _integrate_sections(self, other_section: Section):
        potential_similar_sections = [
            {
                "section": section,
                "similarity": self.similarity_function(
                    self.textbook.section_vectors[section],
                    self.textbook.section_vectors[other_section],
                ),
            }
            for section in self.textbook.textbook.sections
        ]
        best_potential_similar_section = max(
            potential_similar_sections, key=lambda s: s["similarity"]
        )
        if best_potential_similar_section["similarity"] > self.similarity_threshold:
            section_from_this_textbook = best_potential_similar_section["section"]
        else:
            section_from_this_textbook = None
        self.section_mapping[section_from_this_textbook] |= other_section

    def integrate_sections(self, other_textbooks: list[TextbookWithSectionVectors]):
        for textbook in other_textbooks:
            for section in textbook.textbook.sections.values():
                self._integrate_sections(section)


def assign_section_numbers(textbook, sections_dict):
    """Assigns section numbers to top-level sections and their subsections"""
    section_number = 1
    for section_id in textbook.sections:
        if section_id in sections_dict:
            sections_dict[section_id].assign_section_number((section_number,))
            section_number += 1


def build_textbook_hierarchy(sections_dict, data):
    """Use subsection data to populate the subsections attributes"""
    for section_id, section_data in data.items():
        section = sections_dict[section_id]
        section.subsections = [
            sections_dict[sub_id]
            for sub_id in section_data.get("subsections", [])
            if sub_id in sections_dict
        ]


def parse_json_to_textbook(json_file_path: Path) -> Textbook:
    """Parses JSON serialized textbook as Textbook object"""
    with open(json_file_path, encoding="utf-8") as file:
        data = json.load(file)
    textbook = Textbook(json_file_path.stem)

    sections_dict = {}
    for section_id, section_data in data.items():
        new_section = Section(
            section_id=section_id,
            header=section_data["header"],
            content_xml=section_data["content_xml"],
            content_string=section_data["content_string"],
            word_count=section_data["word_count"],
            subsections=section_data.get("subsections", []),
            annotations=section_data["annotations"],
        )
        sections_dict[section_id] = new_section

    build_textbook_hierarchy(sections_dict, data)

    # Add top-level sections to textbook
    for section_id, section in sections_dict.items():
        # Assuming top-level sections are those not listed as a subsection of any other section
        if not any(
            section_id in s_data.get("subsections", []) for s_data in data.values()
        ):
            textbook.add_section(section)

    assign_section_numbers(textbook, sections_dict)

    return textbook
