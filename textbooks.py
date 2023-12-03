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
                for sub_id in section_data["subsections"]
                if sub_id in sections_dict
            ]
            for s in section.subsections:
                s.textbook = self

    def __hash__(self) -> int:
        return hash(self.name)

    def compute_section_vectors(self, section: Optional["Section"] = None):
        """
        Computes and aggregates section vectors.
        If no section is provided, computes for all top-level sections.
        """
        if section is None:
            for top_level_section in self.subsections:
                self.compute_section_vectors(top_level_section)
            return

        weights = []
        for subsection in section.subsections:
            self.compute_section_vectors(subsection)
            weights.append(subsection.word_count)

        if not self.aggregate_subsection_vectors:
            self.section_vectors[section] = self.compute_vector(section)
            return

        vectors = [
            self.section_vectors.get(subsection, 0)
            for subsection in section.subsections
        ]
        this_section_vector = self.compute_vector(section)
        vectors.append(this_section_vector)
        this_section_weight = section.word_count
        weights.append(this_section_weight)

        if sum(weights) > 0:
            aggregated_vector = np.average(vectors, weights=weights)
        else:
            aggregated_vector = None

        self.section_vectors[section] = aggregated_vector


@dataclass()
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
    base_to_other_map: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )
    other_to_base_map: dict[Section, dict[Section, float]] = field(
        default_factory=dict, repr=False, init=False
    )

    def _find_best_matching_section(
        self, other_section_vector
    ) -> tuple[Section | None, float]:
        """Finds the best matching section in the base textbook for a given vector."""
        best_match = max(
            self.base_textbook.all_subsections,
            key=lambda section: self.similarity_function(
                self.base_textbook.section_vectors[section], other_section_vector
            ),
            default=None,
        )
        similarity = self.similarity_function(
            self.base_textbook.section_vectors.get(best_match, 0), other_section_vector
        )
        return {
            "score": similarity,
            "section": best_match if similarity > self.similarity_threshold else None,
        }

    def integrate_sections(self, other_textbooks: list[Textbook]):
        """Integrates similar sections from other_textbooks into the base textbook."""
        for other_textbook in other_textbooks:
            for other_section in other_textbook.all_subsections:
                new_match = self._find_best_matching_section(
                    other_textbook.section_vectors[other_section]
                )
                previous_match = self.other_to_base_map.get(
                    other_section, {"score": -1, "section": None}
                )

                if (
                    previous_match["section"] is not None
                    and new_match["score"] < previous_match["score"]
                ):
                    continue
                self.base_to_other_map[new_match["section"]].add(other_section)
                previous_match_group = self.base_to_other_map[previous_match["section"]]
                if other_section in previous_match_group:
                    previous_match_group.remove(other_section)
                self.other_to_base_map[other_section] = new_match

    def print_matches(self):
        """Prints a textual representation of the base textbook
        with semantic matches from other sections."""
        self.base_textbook.print_toc(self.base_to_other_map)
        print("------------------------------------")
        unmatched_sections = self.base_to_other_map[None]
        if len(unmatched_sections) > 20:
            print(len(unmatched_sections), "unmatched sections")
        else:
            print("Unmatched Sections:")
            for unmatched in unmatched_sections:
                print(f"-\t{unmatched}")
