from computations.doc2vec import cos, doc2vec_vector_computation
from computations.tfidf import sklearn_cos, tfidf_vector_computation
from textbooks.integration import SimilarityBasedTextbookIntegration


def tfidf_doc2vec_integration(
    base_textbook,
    other_textbooks,
    tfidf_text_extraction_fns,
    doc2vec_text_extraction_fn,
    upper_threshold,
    lower_threshold,
    weights=None,
):
    if weights is None:
        weights = [1] * len(tfidf_text_extraction_fns)

    integrated_textbook_tfidf = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        scoring_fn=lambda a, b: sklearn_cos(a, b)[0][0],
        threshold=upper_threshold,
    )
    corpus = integrated_textbook_tfidf.corpus
    tfidf_vectors = tfidf_vector_computation(corpus, tfidf_text_extraction_fns, weights)
    integrated_textbook_tfidf.add_section_vectors(tfidf_vectors)
    integrated_textbook_tfidf.integrate_sections()

    integrated_textbook_doc2vec = SimilarityBasedTextbookIntegration(
        base_textbook=integrated_textbook_tfidf.base_textbook,
        other_textbooks=integrated_textbook_tfidf.other_textbooks,
        scoring_fn=cos,
        threshold=lower_threshold,
    )

    doc2vec_vectors = doc2vec_vector_computation(
        integrated_textbook_doc2vec.corpus, doc2vec_text_extraction_fn
    )
    integrated_textbook_tfidf.add_section_vectors(doc2vec_vectors)
    integrated_textbook_tfidf.integrate_sections()

    integrated_textbook_tfidf.print_matches()
