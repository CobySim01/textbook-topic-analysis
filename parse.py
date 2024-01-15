import re
from copy import copy

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


def get_top_level_text(tag):
    """Extracts the top-level text from the tag."""
    extracted_string = "".join(
        child.string for child in tag.children if isinstance(child, NavigableString)
    )
    return re.sub(r"\s+", " ", extracted_string).strip()


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


def get_subsection_refs(entry):
    """Gets the subsections associated with a TOC entry."""
    next_sibling = entry.find_next_sibling()
    if not next_sibling or next_sibling.name != "list":
        return []
    return [
        ref.attrs["target"]
        for ref in find_all_with_limit(next_sibling, "ref", 2)
        if ref.has_attr("target")
    ]


def remove_subsection_content(content_xml, subsection_refs):
    """Removes content that belongs to subsections."""
    for sub_ref in subsection_refs:
        sub_contents = content_xml.find_all("div", {"xml:id": sub_ref})
        for sub_content in sub_contents:
            sub_content.decompose()


def get_concepts(index, entry_id):
    """Gets the concepts for a given entry."""
    index_refs = index.find_all("ref", attrs={"target": entry_id})
    if index_refs is None:
        return {}
    concepts = {}
    for index_ref in index_refs:
        concept_data = index_ref.parent
        if concept_data.get("domain-specificity") not in {"core-domain", "in-domain"}:
            continue
        concept_id = concept_data.attrs["xml:id"]
        if concept_id.startswith("example") or concept_id.endswith("example"):
            continue
        concepts[concept_id] = {"name": get_top_level_text(concept_data)}
        concept = concept_data.find("seg")
        if concept is None:
            continue
        definition = concept.find("gross", attrs={"property": "skos:definition"})
        if definition is not None:
            concepts[concept_id]["definition"] = definition.text
        subject = concept.find("ref", attrs={"property": "terms:subject"})
        if subject is not None:
            concepts[concept_id]["subject"] = (
                subject.attrs["resource"]
                .removeprefix("http://dbpedia.org/resource/Category:")
                .replace("_", " ")
            )

    return concepts


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
        section_entry = get_top_level_text(entry)
        entry_id = ref.attrs["target"]
        content_xml = copy(body.find("div", attrs={"xml:id": entry_id}))
        subsection_refs = get_subsection_refs(entry)
        if content_xml:
            level = int(content_xml.attrs["n"])
            remove_subsection_content(content_xml, subsection_refs)
            content = convert_xml_content_to_string(content_xml)
            word_count = len(content_xml.find_all("w"))
            concepts = get_concepts(index, entry_id)
            toc_entries[entry_id] = {
                "entry": section_entry,
                "level": level,
                "content": content,
                "word_count": word_count,
                "subsections": subsection_refs,
                "concepts": concepts,
            }
    return toc_entries
