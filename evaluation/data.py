from pathlib import Path
from statistics import mean
from typing import Optional

import pandas as pd

from textbooks.data import Section, Textbook

BOOK_COLUMNS = ["section_id", "title"]


def find_section(header: str, textbook: Textbook = None) -> Optional[Section]:
    """Finds a section in Textbook by the full text representation of its header."""
    if textbook is None:
        raise ValueError
    for section in textbook.all_subsections():
        section_number_string = ".".join(str(i) for i in section.section_number) + " "
        if header.startswith(section_number_string) or section.header == header:
            return section
    return None


def parse_mapping_file():
    EVALUATION_DATA_DIR = Path("textbooks-data/evaluation")
    REPEATED_COLUMNS = ["expert", "confidence", "relevance", "score"]
    MAPPING_BASE_COLUMNS = ["base_section_id", "other_section_id"]
    MAPPING_COLUMNS = MAPPING_BASE_COLUMNS + [
        f"{col}{i}" for i in range(3) for col in REPEATED_COLUMNS
    ]
    mapping = pd.read_csv(
        EVALUATION_DATA_DIR / "1_mapping.csv", sep=";", names=MAPPING_COLUMNS
    )

    sub_dfs = [
        mapping[
            MAPPING_BASE_COLUMNS + [f"{col}{i}" for col in REPEATED_COLUMNS]
        ].rename(columns={f"{col}{i}": col for col in REPEATED_COLUMNS})
        for i in range(3)
    ]
    long_mapping = pd.concat(sub_dfs).dropna()
    long_mapping[["confidence", "relevance", "score"]] = long_mapping[
        ["confidence", "relevance", "score"]
    ].astype("int")
    long_mapping = (
        long_mapping.groupby(MAPPING_BASE_COLUMNS)
        .apply(
            lambda group: group.drop(columns=MAPPING_BASE_COLUMNS).to_dict("records")
        )
        .reset_index(name="data")
    )

    new_attributes = [
        ("num_experts", len),
        ("score_6_and_above", lambda data: len([x for x in data if x["score"] >= 6])),
        ("score_9", lambda data: len([x for x in data if x["score"] == 9])),
        ("mean_score", lambda data: mean(x["score"] for x in data)),
        ("mean_confidence", lambda data: mean(x["confidence"] for x in data)),
        ("mean_relevance", lambda data: mean(x["relevance"] for x in data)),
        ("relevance_3", lambda data: len([x for x in data if x["relevance"] == 3])),
    ]
    for attribute_name, function in new_attributes:
        long_mapping[attribute_name] = long_mapping.data.apply(function)
    return long_mapping


def get_expert_mapping(
    base_textbook_object, base_textbook_file, other_textbook_object, other_textbook_file
):
    base_textbook_sections = pd.read_csv(
        base_textbook_file, sep=";", names=BOOK_COLUMNS
    )
    base_textbook_sections["section_obj"] = base_textbook_sections.title.apply(
        lambda t: find_section(t, textbook=base_textbook_object)
    )
    base_textbook_sections = base_textbook_sections.add_prefix("base_")

    other_textbook_sections = pd.read_csv(
        other_textbook_file, sep=";", names=BOOK_COLUMNS
    )
    other_textbook_sections["section_obj"] = other_textbook_sections.title.apply(
        lambda t: find_section(t, textbook=other_textbook_object)
    )
    other_textbook_sections = other_textbook_sections.add_prefix("other_")

    long_mapping = parse_mapping_file()
    long_mapping = (
        base_textbook_sections.merge(long_mapping, how="left", on="base_section_id")
        .astype({"other_section_id": "Int64"})
        .merge(other_textbook_sections, how="left", on="other_section_id")
    )

    columns_to_remove = {
        "base_section_id",
        "base_title",
        "other_section_id",
        "other_title",
    }
    long_mapping = (
        long_mapping[long_mapping.base_section_obj.notnull()][
            [c for c in long_mapping.columns if c not in columns_to_remove]
        ]
        .groupby("base_section_obj", sort=False)
        .apply(lambda x: x.to_dict(orient="records"))
        .to_dict()
    )
    return {
        key: {
            value["other_section_obj"]: {
                k: v for k, v in value.items() if k != "other_section_obj"
            }
            for value in values
            if isinstance(value["data"], list)
        }
        for key, values in long_mapping.items()
    }
