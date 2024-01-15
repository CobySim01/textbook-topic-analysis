import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from computation.doc2vec import doc2vec_integration
from textbooks.data import Textbook
from textbooks.utils import extract_content


def integrate_textbook(base_textbook_path, all_textbook_paths):
    """Function to integrate a single textbook"""
    print(f"Integrating with base {base_textbook_path.stem}...")
    result = (
        base_textbook_path.stem,
        doc2vec_integration(
            base_textbook=Textbook.from_json(base_textbook_path),
            other_textbooks=tuple(
                Textbook.from_json(t)
                for t in all_textbook_paths
                if t != base_textbook_path
            ),
            text_extraction_fn=extract_content,
            threshold=0.4,
            vector_size=100,
            min_count=1,
            epochs=40,
            iterative=False,
            evaluate=False,
        ).dataset,
    )
    print(f"Finished integrating with base {base_textbook_path.stem}.")
    return result


if __name__ == "__main__":
    all_textbook_paths = list(Path("textbooks-parsed").glob("*"))
    datasets = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                integrate_textbook, base_textbook_path, all_textbook_paths
            ): base_textbook_path
            for base_textbook_path in all_textbook_paths
        }
        for future in as_completed(futures):
            base_textbook, result = future.result()
            datasets[base_textbook] = result

    with open("datasets.json", "w", encoding="utf-8") as f:
        json.dump(datasets, f)
