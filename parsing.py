import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString


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
        content = body.find("div", attrs={"xml:id": entry_id})
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


if __name__ == "__main__":
    for file in Path("textbooks").glob("*.xml"):
        parse_file(file)
