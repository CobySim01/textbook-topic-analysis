from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Optional

import numpy as np


@dataclass
class Textbook:
    """Represents a Textbook document."""

    name: str
    subsections: list["Section"] = field(default_factory=list, repr=False, init=False)
    compute_vector: Callable[["Section"], Any] = field(default=None, repr=False)
    aggregate_subsection_vectors: bool = field(default=False, repr=False)
    section_vectors: dict["Section", Any] = field(
        default_factory=dict, repr=False, init=False
    )

    def add_section(self, section: "Section"):
        """Adds a section to this textbook's sections"""
        self.subsections.append(section)
        section.textbook = self

    def print_toc(self, matches=None):
        """Prints a textual representation of the Textbook's table of contents."""
        for section in self.subsections:
            section.print_entry()
            if matches is not None:
                for match in matches[section]:
                    print(f"-\t{match}")
            section.print_subsections(matches=matches, indent="\t")

    @property
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
                for sub_id in section_data.get("subsections", [])
                if sub_id in sections_dict
            ]
            for s in section.subsections:
                s.textbook = self

    def __hash__(self) -> int:
        return hash(self.name)

    def compute_section_vectors(self):
        """Computes and aggregates section vectors."""
        for section in self.subsections:
            self._compute_section_vectors(section)

    def _compute_section_vectors(self, section: "Section"):
        """Computes and aggregates section vectors."""
        weights = []
        for subsection in section.subsections:
            self._compute_section_vectors(subsection)
            weights.append(subsection.word_count)
        if not self.aggregate_subsection_vectors:
            self.section_vectors |= {section: self.compute_vector(section)}
            return
        vectors = [
            self.section_vectors[subsection] for subsection in section.subsections
        ]
        this_section_vector = self.compute_vector(section)
        vectors.append(this_section_vector)
        this_section_weight = section.word_count
        weights.append(this_section_weight)
        vectors_without_nulls = [v if v is not None else 0 for v in vectors]
        if sum(weights) > 0:
            aggregated_vector = np.average(vectors_without_nulls, weights=weights)
        else:
            aggregated_vector = None
        self.section_vectors |= {section: aggregated_vector}
        return


@dataclass()
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

    def print_subsections(self, matches=None, indent=""):
        """Prints the subsections for a textbook"""
        for section in self.subsections:
            section.print_entry(indent)
            if matches is not None:
                for match in matches[self]:
                    print(f"{indent}-\t{match}")
            section.print_subsections(matches, indent + "\t")

    def __hash__(self) -> int:
        return hash((self.textbook, self.section_id, self.header, self.content_string))


@dataclass
class IntegratedTextbook:
    """Represents a Textbook integrated with sections from other textbooks."""

    base_textbook: Textbook
    similarity_function: Callable[[str, str], float]
    similarity_threshold: float
    section_mapping: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )

    def _integrate_sections(self, other_textbook: Textbook, other_section: Section):
        potential_similar_sections = [
            {
                "section": section,
                "similarity": self.similarity_function(
                    self.base_textbook.section_vectors[section],
                    other_textbook.section_vectors[other_section],
                ),
            }
            for section in self.base_textbook.all_subsections
        ]
        best_potential_similar_section = max(
            potential_similar_sections, key=lambda s: s["similarity"]
        )
        if best_potential_similar_section["similarity"] > self.similarity_threshold:
            section_from_this_textbook = best_potential_similar_section["section"]
        else:
            section_from_this_textbook = None
        self.section_mapping[section_from_this_textbook].add(other_section)

    def integrate_sections(self, other_textbooks: list[Textbook]):
        """Integrates similar sections from other_textbooks into the base textbook."""
        for textbook in other_textbooks:
            for section in textbook.all_subsections:
                self._integrate_sections(textbook, section)

    def print_matches(self):
        """Prints a textual representation of the base textbook
        with semantic matches from other sections."""
        self.base_textbook.print_toc(self.section_mapping)
        print("------------------------------------")
        unmatched_sections = self.section_mapping[None]
        if len(unmatched_sections) > 20:
            print(len(unmatched_sections), "unmatched sections")
        else:
            print("Unmatched Sections:")
            for unmatched in unmatched_sections:
                print(f"-\t{unmatched}")
