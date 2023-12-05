from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, DefaultDict, Iterable, Optional
import json


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
    # content_xml: str = field(compare=False, repr=False)
    content_string: str = field(compare=False, repr=False)
    word_count: int = field(compare=False, repr=False)
    subsections: list["Section"] = field(compare=False, repr=False)
    annotations: list[str] = field(compare=False, repr=False)
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
        return hash((self.textbook, self.section_id, self.header, self.content_string))


@dataclass
class IntegratedTextbook:
    """Represents a Textbook integrated with sections from other textbooks."""

    base_textbook: Textbook
    other_textbooks: list[Textbook]

    similarity_function: Callable[[str, str], float] = field(repr=False)
    similarity_threshold: float = field(repr=False)
    vectors: dict[Section, Any] = field(default_factory=dict, repr=False, init=False)

    base_to_other_map: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )
    other_to_base_map: dict[Section, dict[Section, float]] = field(
        default_factory=dict, repr=False, init=False
    )

    sections_to_integrate: Optional[Iterable[Section]] = field(
        default=None, repr=False, init=False
    )

    @property
    def corpus(self):
        """Returns the corpus associated with this IntegratedTextbook"""
        return [
            section
            for textbook in [self.base_textbook] + self.other_textbooks
            for section in textbook.all_subsections()
        ]

    def add_section_vectors(self, section_vectors_map: dict["Section", Any]):
        """Add section vectors to this Textbook"""
        self.vectors |= section_vectors_map

    def _find_best_matching_section(
        self, other_section
    ) -> tuple[Section | None, float]:
        """Finds the best matching section in the base textbook for a given vector."""
        other_section_vector = self.vectors[other_section]
        best_match = max(
            self.base_textbook.all_subsections(),
            key=lambda section: self.similarity_function(
                self.vectors[section], other_section_vector
            ),
            default=None,
        )
        similarity = self.similarity_function(
            self.vectors.get(best_match, 0), other_section_vector
        )
        return {
            "score": similarity,
            "section": best_match if similarity > self.similarity_threshold else None,
        }

    def integrate_sections(self):
        """Attempts to integrate all sections from other_textbooks into the base textbook."""
        if self.sections_to_integrate is not None:
            raise ValueError("Cannot use (non-)/iterative approaches together")
        for other_textbook in self.other_textbooks:
            for other_section in other_textbook.all_subsections():
                self._integrate_section(other_section)

    def integrate_section(self):
        """Attempts to integrate a single section from other_textbooks into the base
        textbook, taking the next section from `self.sections_to_integrate`."""
        if self.sections_to_integrate is None:
            self.sections_to_integrate = (
                other_section
                for other_textbook in self.other_textbooks
                for other_section in other_textbook.all_subsections()
            )
        section = next(self.sections_to_integrate)
        self._integrate_section(section)

    def _integrate_section(self, section):
        """Integrates a section from `other_textbooks`"""
        new_match = self._find_best_matching_section(section)

        if section in self.other_to_base_map:
            old_match = self.other_to_base_map[section]
            if old_match["score"] > new_match["score"]:
                return
            if old_match["section"] is None and new_match["section"] is None:
                return
            self.base_to_other_map[old_match["section"]].remove(section)

        self.base_to_other_map[new_match["section"]].add(section)
        self.other_to_base_map[section] = new_match

    def print_matches(self):
        """Prints a textual representation of the base textbook
        with semantic matches from other sections."""
        print_toc(self.base_textbook, self.base_to_other_map)
        print("------------------------------------")
        unmatched_sections = self.base_to_other_map[None]
        print(len(unmatched_sections), "unmatched sections")


def print_toc(section: Section | Textbook, matches: dict = None, indent: str = ""):
    """Prints a textual representation of a section's table of contents."""
    if isinstance(section, Section):
        section.print_entry(indent)
        if matches is not None and section in matches:
            for match in matches[section]:
                print(f"{indent}-\t{match}")
        indent += "\t"
    for subsection in section.subsections:
        print_toc(subsection, matches, indent)
