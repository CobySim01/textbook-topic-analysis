from functools import lru_cache, reduce

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

from nlp import lemmatize
from textbooks.integration import SimilarityBasedTextbookIntegration


def compose(*functions):
    """Composes the passed functions (with the first function provided applied last)."""
    return reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), functions)


def tfidf_vector_computation(corpus, text_extraction_fns, weights):
    # Flatten sections for each text extraction function
    flattened_sections = [
        [fn(section) for section in corpus] for fn in text_extraction_fns
    ]

    # Combine and vectorize the flattened sections
    combined_sections = [
        section for fn_sections in flattened_sections for section in fn_sections
    ]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_sections)

    # Split the TF-IDF matrix for each text extraction function
    split_indices = [len(fn_sections) for fn_sections in flattened_sections]
    cumulative_indices = np.insert(arr=np.cumsum(split_indices), obj=0, values=0)
    split_matrices = [
        tfidf_matrix[cumulative_indices[i] : cumulative_indices[i + 1]]
        for i in range(len(text_extraction_fns))
    ]

    # Compute weighted average of vectors for each section
    section_vectors = {}
    for i, section in enumerate(corpus):
        vectors = [split_matrices[j][i] for j in range(len(text_extraction_fns))]
        section_vectors[section] = np.average(vectors, weights=weights)
    return section_vectors


@lru_cache(maxsize=None)
def tfidf_integration(
    base_textbook,
    other_textbooks,
    iterative,
    text_extraction_fns,
    threshold,
    preprocessing=None,
    weights=None,
    evaluate=True,
):
    if weights is None:
        weights = [1] * len(text_extraction_fns)

    if preprocessing == "lemmatize":
        text_extraction_fns = [
            compose(" ".join, lemmatize, fn) for fn in text_extraction_fns
        ]

    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        scoring_fn=lambda a, b: sklearn_cos(a, b)[0][0],
        threshold=threshold,
        iterative=iterative,
    )
    corpus = integrated_textbook.corpus
    vectors = tfidf_vector_computation(corpus, text_extraction_fns, weights)
    integrated_textbook.add_section_vectors(vectors)

    if iterative:
        for match in integrated_textbook.integrate_sections():
            if match is None:
                continue
            base_section, other_section = match
            base_section.extend(other_section)
            vectors = tfidf_vector_computation(corpus, text_extraction_fns, weights)
            integrated_textbook.add_section_vectors(vectors)
    else:
        integrated_textbook.integrate_sections()
    if not evaluate:
        return integrated_textbook
    integrated_textbook.print_matches()
    return integrated_textbook.evaluate()
