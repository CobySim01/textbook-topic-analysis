from pathlib import Path


import pandas as pd


BOOK_COLUMNS = ["section_id", "title"]


def parse_mapping_file():
    EVALUATION_DATA_DIR = Path("evaluation-data")
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

    return long_mapping


def get_expert_mapping(base_textbook_file, other_textbook_file):
    base_textbook_sections = pd.read_csv(
        base_textbook_file, sep=";", names=BOOK_COLUMNS
    )
    base_textbook_sections = base_textbook_sections.add_prefix("base_")

    other_textbook_sections = pd.read_csv(
        other_textbook_file, sep=";", names=BOOK_COLUMNS
    )
    other_textbook_sections = other_textbook_sections.add_prefix("other_")

    long_mapping = parse_mapping_file()
    long_mapping = (
        base_textbook_sections.merge(long_mapping, how="left", on="base_section_id")
        .astype({"other_section_id": "Int64"})
        .merge(other_textbook_sections, how="left", on="other_section_id")
    )
    long_mapping = long_mapping[long_mapping.other_section_id.notnull()]
    long_mapping.to_json("evaluation/expert-mapping.json", orient="records")


get_expert_mapping(
    "evaluation-data/1_book2_sections.csv", "evaluation-data/1_book1_sections.csv"
)
