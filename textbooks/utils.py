import re


def extract_section_number(entry):
    """Extracts the section header for a given TOC entry."""
    section_number_match = re.search(r"\b(\w|\d+)(\.\d+)*\b", entry)
    if section_number_match is not None:
        return section_number_match.group()
    return None


def section_number_string_to_tuple(string):
    if string is None:
        return None
    section_number_list = []
    for part in string.split("."):
        try:
            section_number_list.append(int(part))
        except ValueError:
            section_number_list.append(part)
    return tuple(section_number_list)


def remove_section_number(entry):
    section_number = extract_section_number(entry)
    if section_number is None:
        return entry
    return entry.replace(section_number, "").strip()


def is_valid_entry(entry: str) -> bool:
    """Determines whether a section entry is removed from processing."""
    HEADER_CANNOT_CONTAIN = [
        "exercises",
        "solutions",
        "index",
        "glossary",
        "references",
        "appendix",
    ]
    HEADER_CANNOT_START_WITH = ["a."]
    HEADER_CANNOT_EQUAL = {"introduction"}
    return not (
        entry in HEADER_CANNOT_EQUAL
        or any(substring in entry for substring in HEADER_CANNOT_CONTAIN)
        or any(entry.startswith(prefix) for prefix in HEADER_CANNOT_START_WITH)
    )


def extract_header(section):
    return section.header


def extract_content(section):
    return section.content


def extract_concept_name(section):
    return " ".join(c["name"] for c in section.concepts.values())


def extract_concept_definition(section):
    return " ".join(
        c["definition"] for c in section.concepts.values() if "definition" in c
    )


def extract_concept_subject(section):
    return " ".join(c["subject"] for c in section.concepts.values() if "subject" in c)
