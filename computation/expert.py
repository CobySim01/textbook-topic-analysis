import nltk
from evaluation.expert import get_expert_mapping


from textbooks.integration import TextbookIntegration

nltk.download("punkt", quiet=True)


def expert_integration(base_textbook, other_textbooks):
    integrated_textbook = TextbookIntegration(
        base_textbook=base_textbook, other_textbooks=other_textbooks
    )
    integrated_textbook.base_to_other_map = get_expert_mapping(
        base_textbook, other_textbooks[0]
    )
    return integrated_textbook
