import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Optional, TypedDict

from evaluation.data import get_expert_mapping

from .data import Section, Textbook

MatchingSection = TypedDict(
    "MatchingSection", {"score": float, "section": Optional[Section]}
)
SimilarityFunction = Callable[[Section, Section], float]
QueryFunction = Callable[[Section], float]


@dataclass(kw_only=True)
class TextbookIntegration(ABC):
    """Represents a Textbook integrated with sections from other textbooks."""

    base_textbook: Textbook
    other_textbooks: list[Textbook]

    scoring_fn: Optional[SimilarityFunction | QueryFunction] = field(
        default=None, repr=False
    )
    threshold: Optional[float] = field(default=None, repr=False)

    # The set of sections that map to each base section
    base_to_other_map: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )
    # Ensure that each other section can only map to one base section
    many_to_many: bool = field(default=True)
    _other_to_base_map: dict[Section, MatchingSection] = field(
        default_factory=dict, repr=False, init=False
    )

    @property
    def corpus(self):
        """Returns the corpus associated with this IntegratedTextbook"""
        return [
            section
            for textbook in [self.base_textbook] + self.other_textbooks
            for section in textbook.all_subsections()
        ]

    @abstractmethod
    def find_best_matching_section(self, other_section: Section) -> MatchingSection:
        """Finds the best matching section in the base textbook for a given vector."""

    def integrate_sections(self):
        """Attempts to integrate all sections from other_textbooks into the base textbook."""
        for other_textbook in self.other_textbooks:
            for other_section in other_textbook.all_subsections():
                self._integrate(other_section)

    def _integrate(self, section):
        """Integrates a section from `other_textbooks`"""
        new_match = self.find_best_matching_section(section)

        if not self.many_to_many and section in self._other_to_base_map:
            old_match = self._other_to_base_map[section]
            if old_match["score"] > new_match["score"]:
                return
            if old_match["section"] is None and new_match["section"] is None:
                return
            self.base_to_other_map[old_match["section"]].remove(section)

        self.base_to_other_map[new_match["section"]].add(section)
        if not self.many_to_many:
            self._other_to_base_map[section] = new_match

    def print_matches(self):
        """Prints a textual representation of the base textbook
        with semantic matches from other sections."""
        print_toc(self.base_textbook, self.base_to_other_map)
        print("------------------------------------")
        unmatched_sections = self.base_to_other_map[None]
        print(len(unmatched_sections), "unmatched sections")

    def evaluate(self):
        """Returns summary statistics for the integrated textbok."""
        if len(self.other_textbooks) != 1:
            raise ValueError("Cannot evaluate for more than one other textbooks.")
        expert_mapping = get_expert_mapping(self.base_textbook, self.other_textbooks[0])

        # Where algorithm correctly identifies similar sections (agreement with experts).
        true_positives = 0
        # When algorithm incorrectly identifies sections as similar (disagreement with experts).
        false_positives = 0
        # Sections that your algorithm fails to identify as similar, but experts identify as similar.
        false_negatives = 0

        for base_section in self.base_textbook.all_subsections():
            expert_mapped = set(expert_mapping.get(base_section, {}))
            algorithm_mapped = self.base_to_other_map[base_section]
            true_positives += len(expert_mapped & algorithm_mapped)
            false_positives += len(algorithm_mapped - expert_mapped)
            false_negatives += len(expert_mapped - algorithm_mapped)

        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = None

        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = None

        if true_positives > 0 and precision is not None and recall is not None:
            f1 = 2 / ((1 / precision) + (1 / recall))
        else:
            f1 = None

        def extract_all_mappings(mapping):
            return set(
                (base_section, other_section)
                for base_section, other_sections in mapping.items()
                for other_section in other_sections
            )

        all_expert_mappings = extract_all_mappings(expert_mapping)
        all_algorithm_mappings = extract_all_mappings(self.base_to_other_map)

        # cardinality of intersection divided by cardinality of union
        jaccard_index = len(all_expert_mappings & all_algorithm_mappings) / len(
            all_expert_mappings | all_algorithm_mappings
        )

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard_index": jaccard_index,
        }


@dataclass(kw_only=True)
class SimilarityBasedTextbookIntegration(TextbookIntegration):
    """Represents a Textbook integrated using similarity-based methods."""

    vectors: dict[Section, Any] = field(default_factory=dict, repr=False, init=False)

    def add_section_vectors(self, section_vectors_map: dict[Section, Any]):
        """Add section vectors to this Textbook"""
        self.vectors |= section_vectors_map

    def find_best_matching_section(self, other_section: Section) -> MatchingSection:
        if self.vectors:
            other_section = self.vectors[other_section]

        def get_section_vector(section):
            return self.vectors[section] if self.vectors else section

        section_similarity_scores_iterable = (
            (section, self.scoring_fn(get_section_vector(section), other_section))
            for section in self.base_textbook.all_subsections()
        )
        best_match, best_match_score = max(
            section_similarity_scores_iterable, key=lambda x: x[1], default=(None, None)
        )

        if best_match_score > self.threshold:
            returned_best_match = best_match
        else:
            returned_best_match = None

        return {"score": best_match_score, "section": returned_best_match}


@dataclass(kw_only=True)
class QueryBasedTextbookIntegration(TextbookIntegration):
    """Represents a Textbook integrated using query-based methods."""

    def find_best_matching_section(self, other_section: Section) -> MatchingSection:
        result = self.scoring_fn(other_section)
        if result["score"] > self.threshold:
            section = result["section"]
        else:
            section = None
        return {"score": result["score"], "section": section}


def print_toc(
    section: Section | Textbook, matches: Optional[dict] = None, indent: str = ""
):
    """Prints a textual representation of a section's table of contents."""
    if isinstance(section, Section):
        section.print_entry(indent)
        if matches is not None and section in matches:
            for match in matches[section]:
                print(f"{indent}-\t{match}")
        indent += "\t"
    for subsection in section.subsections:
        print_toc(subsection, matches, indent)
