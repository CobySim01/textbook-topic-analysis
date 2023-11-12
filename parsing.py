import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag


def join_hyphenated_words(words):
    HYPHEN = "-"
    i = 0
    while i < len(words) - 1:
        if words[i].endswith(HYPHEN):
            words[i] = words[i].rstrip(HYPHEN) + words[i + 1]
            del words[i + 1]
        else:
            i += 1
    return words


def convert_raw_content_to_paragraphs(raw_content: Tag):
    content = []
    for child in raw_content.find_all("ab", attrs={"type": "Body"}):
        for grandchild in child.children:
            if not grandchild.text.strip():
                continue
            if grandchild.name == "w":
                content += [text.strip() for text in grandchild.stripped_strings]
    return " ".join(join_hyphenated_words(content))


def parse_file(path: Path) -> dict:
    """
    Parses a TEI-encoded XML file into a dictionary of TOC entries -> section contents
    """
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, features="xml")

    toc = soup.find("front").find("div", attrs={"type": "contents"}).find("list")
    body = soup.find("body")

    toc_entries = []
    for entry in toc.find_all("item"):
        entry_text = "".join(
            child for child in entry.contents if isinstance(child, NavigableString)
        ).strip()
        section_number_match = re.search(r"\b\d+(\.\d+)*\b", entry_text)
        if section_number_match is not None:
            section_number = section_number_match.group()
            header = entry_text.replace(section_number, "").strip()
        else:
            section_number = None
        entry_id = entry.find("ref").attrs["target"]
        raw_content = body.find("div", attrs={"xml:id": entry_id})
        if raw_content is not None:
            content = convert_raw_content_to_paragraphs(raw_content)
        else:
            content = None
        toc_entries.append(
            {
                "id": entry_id,
                "header": header,
                "section_number": section_number,
                "span": tuple(int(num.text) for num in entry.find_all("num")),
                "content": content,
            }
        )
    return toc_entries
