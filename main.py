from pathlib import Path
from parsing import parse_file


if __name__ == "__main__":
    for file in Path("textbooks").glob("*.xml"):
        parsed_file_contents = parse_file(file)
