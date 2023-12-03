import json
import re
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag


def join_hyphenated_words(words: list[str]):
    """Joins words that have been split as a result of end-of-line hyphenation."""
    HYPHEN = "-"
    i = 0
    while i < len(words) - 1:
        if words[i].endswith(HYPHEN):
            words[i] = words[i].rstrip(HYPHEN) + words[i + 1]
            del words[i + 1]
        else:
            i += 1
    return words


def convert_xml_content_to_string(raw_content: Tag):
    """Converts the XML content of textbook section to a string"""
    content = []
    for child in raw_content.find_all("ab", attrs={"type": "Body"}):
        for grandchild in child.children:
            if not grandchild.text.strip():
                continue
            if grandchild.name == "w":
                content += [text.strip() for text in grandchild.stripped_strings]
    return " ".join(join_hyphenated_words(content))


def find_all_with_limit(soup, tag, max_depth, current_depth=0):
    """Alternative to `soup.find_all(tag, recursive=True)` that puts a
    limit on recursive search depth."""
    found_items = []
    # Check if the maximum depth has been exceeded
    if current_depth > max_depth:
        return found_items
    # Check if the current element matches the search criteria
    if isinstance(soup, Tag):
        if soup.name == tag:
            found_items.append(soup)
        # Recursively search in child elements
        for child in soup.children:
            found_items.extend(
                find_all_with_limit(child, tag, max_depth, current_depth + 1)
            )
    return found_items


def get_section_header(entry):
    entry_text = "".join(
        child for child in entry.contents if isinstance(child, NavigableString)
    ).strip()
    section_number_match = re.search(r"\b\d+(\.\d+)*\b", entry_text)
    if section_number_match is None:
        return entry_text
    section_number = section_number_match.group()
    return entry_text.replace(section_number, "").strip()


def get_subsection_refs(next_sibling):
    if not next_sibling or next_sibling.name != "list":
        return []
    return [
        ref.attrs["target"]
        for ref in find_all_with_limit(next_sibling, "ref", 2)
        if ref.has_attr("target")
    ]


def remove_subsection_content(content_xml, subsection_refs):
    for sub_ref in subsection_refs:
        sub_contents = content_xml.find_all("div", {"xml:id": sub_ref})
        for sub_content in sub_contents:
            sub_content.decompose()


def get_annotations(index, entry_id):
    """Gets the annotations for a given entry."""
    index_refs = index.find_all("ref", attrs={"target": entry_id})
    if index_refs is None:
        return []
    index_items = " ".join(
        gross
        for ref in index_refs
        for gross in set(ref.parent.find_all("gross"))
        if ref.parent.get("domain-specificity") in {"core-domain", "in-domain"}
    )
    return index_items


def parse_xml(soup: BeautifulSoup) -> dict:
    """
    Parses TEI-encoded XML into a dictionary of TOC entries -> section contents
    """
    toc = soup.find("front").find("div", attrs={"type": "contents"}).find("list")
    body = soup.find("body")
    index = soup.find("div", attrs={"type": "index"})
    toc_entries = {}
    for entry in toc.find_all("item"):
        ref = entry.find("ref")
        if not ref or not ref.has_attr("target"):
            continue
        header = get_section_header(entry)
        entry_id = ref.attrs["target"]
        content_xml = copy(body.find("div", attrs={"xml:id": entry_id}))
        subsection_refs = get_subsection_refs(entry.find_next_sibling())
        if content_xml:
            remove_subsection_content(content_xml, subsection_refs)
            content_string = convert_xml_content_to_string(content_xml)
            word_count = len(content_xml.find_all("w"))
            annotations = get_annotations(index, entry_id)
            toc_entries[entry_id] = {
                "header": header,
                "content_xml": content_xml,
                "content_string": content_string,
                "word_count": word_count,
                "subsections": subsection_refs,
                "annotations": annotations,
            }
    return toc_entries


class SoupJSONEncoder(json.JSONEncoder):
    """Encode bs4.element.Tag as string when serialising JSON."""

    def default(self, o):
        if isinstance(o, Tag):
            return str(o)
        return super().default(o)


def process_single_file(file_pair):
    source_file, output_file = file_pair
    with open(source_file, encoding="utf-8") as f:
        source_xml = BeautifulSoup(f, features="xml")
    parsed_file = parse_xml(source_xml)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_file, f, cls=SoupJSONEncoder, indent=2)


def process_files(file_mapping: dict[Path, Path]):
    with ProcessPoolExecutor() as executor:
        executor.map(process_single_file, file_mapping.items())
