import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Optional, TypedDict

from evaluation.expert import get_expert_mapping

from .data import Section, Textbook

random.seed(2024)


MatchingSection = TypedDict("MatchingSection", {"score": float, "section": Section})
SimilarityFunction = Callable[[Section, Section], float]
QueryFunction = Callable[[Section], float]


def filter_by_section(scores, section: Section):
    """Filters the matrix to only return entries relevant to a given Section."""
    return [
        ([s for s in key if s != section][0], value)
        for key, value in scores.items()
        if section in key
    ]


def random_shuffle(lst):
    """Creates a new list by randomly shuffling an existing list."""
    new_list = list(lst)
    random.shuffle(new_list)
    return new_list


@dataclass(kw_only=True)
class TextbookIntegration:
    """Represents a Textbook integrated with sections from other textbooks."""

    base_textbook: Textbook
    other_textbooks: list[Textbook]

    vectors: dict[Section, Any] = field(default_factory=dict, repr=False, init=False)

    scoring_fn: Optional[SimilarityFunction | QueryFunction] = field(
        default=None, repr=False
    )
    threshold: Optional[float] = field(default=None, repr=False)
    iterative: bool = field(default=False, repr=False)

    base_to_other_map: DefaultDict[Optional[Section], set[Section]] = field(
        default_factory=lambda: defaultdict(set), repr=False, init=False
    )
    scores: dict[tuple[Section, Section], float] = field(
        default_factory=dict, repr=False, init=False
    )

    @property
    def corpus(self):
        """Returns the corpus associated with this IntegratedTextbook"""
        return self.base_textbook.all_subsections() + [
            section
            for textbook in self.other_textbooks
            for section in textbook.all_subsections()
        ]

    @property
    def dataset(self):
        """Returns the dataset created by integrating textbooks."""

        def get_section_data(section: Section, base_section: Optional[Section] = None):
            label_section = section if base_section is None else base_section
            return {
                "topic": label_section.find_ancestor(label_section.textbook).header,
                "subtopic": section.header,
                "content": section.content,
                "concepts": [concept["name"] for concept in section.concepts.values()],
                "textbook": section.textbook.name,
            }

        base_sections = [
            get_section_data(section)
            for section in self.base_to_other_map
            if section is not None and section.is_valid
        ]
        other_sections = [
            get_section_data(other_section, base_section)
            for base_section, other_sections in self.base_to_other_map.items()
            for other_section in other_sections
            if other_section.is_valid
        ]
        return base_sections + other_sections

    def find_best_matching_section(self, other_section: Section) -> MatchingSection:
        """Finds the best matching section in the base textbook for a given vector."""
        for base_section in self.base_textbook.all_subsections():
            self.scores[(base_section, other_section)] = self.scoring_fn(
                self.vectors[base_section] if self.vectors else base_section,
                self.vectors[other_section] if self.vectors else other_section,
            )
        best_match, best_match_score = max(
            filter_by_section(self.scores, other_section),
            key=lambda x: x[1],
            default=(None, None),
        )
        return {"score": best_match_score, "section": best_match}

    def integrate_sections(self):
        """Attempts to integrate all sections from other_textbooks into the base textbook."""
        if self.iterative:
            return (
                self._integrate(other_section)
                for other_textbook in random_shuffle(self.other_textbooks)
                for other_section in random_shuffle(other_textbook.all_subsections())
            )
        for other_textbook in self.other_textbooks:
            for other_section in other_textbook.all_subsections():
                self._integrate(other_section)
        return None

    def _integrate(self, section):
        """Integrates a section from `other_textbooks`"""
        new_match = self.find_best_matching_section(section)
        if new_match["score"] < self.threshold:
            new_match["section"] = None
        self.base_to_other_map[new_match["section"]].add(section)

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
        # Sections that algorithm fails to identify as similar, but experts identify as similar.
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

    def add_section_vectors(self, section_vectors_map: dict[Section, Any]):
        """Add section vectors to this Textbook"""
        self.vectors |= section_vectors_map


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
