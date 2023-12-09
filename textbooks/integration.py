from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Iterable, Optional

from utils import num_arguments

from .data import Section, Textbook


@dataclass
class IntegratedTextbook:
    """Represents a Textbook integrated with sections from other textbooks."""

    base_textbook: Textbook
    other_textbooks: list[Textbook]

    similarity_fn: Callable[..., float] = field(repr=False)
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

    def _find_best_matching_section(self, other_section: Section):
        """Finds the best matching section in the base textbook for a given vector."""
        if not self.vectors and num_arguments(self.similarity_fn) == 1:
            result = self.similarity_fn(other_section)
            if result["score"] > self.similarity_threshold:
                section = result["section"]
            else:
                section = None
            return {"score": result["score"], "section": section}

        if not self.vectors and num_arguments(self.similarity_fn) == 2:
            best_match = max(
                self.base_textbook.all_subsections(),
                key=lambda section: self.similarity_fn(section, other_section),
                default=None,
            )
            similarity = self.similarity_fn(best_match, other_section)
            return {
                "score": similarity,
                "section": best_match
                if similarity > self.similarity_threshold
                else None,
            }

        other_section_vector = self.vectors[other_section]
        best_match = max(
            self.base_textbook.all_subsections(),
            key=lambda section: self.similarity_fn(
                self.vectors[section], other_section_vector
            ),
            default=None,
        )
        similarity = self.similarity_fn(
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
