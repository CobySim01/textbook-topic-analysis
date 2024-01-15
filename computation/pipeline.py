from computation.doc2vec import doc2vec_integration
from computation.tfidf import tfidf_integration


def pipeline_integration(
    base_textbook,
    other_textbooks,
    tfidf_text_extraction_fns,
    tfidf_threshold,
    tfidf_uncertain_threshold,
    d2v_text_extraction_fn,
    d2v_threshold,
    d2v_vector_size,
    d2v_min_count,
):
    tfidf_it = tfidf_integration(
        base_textbook,
        other_textbooks,
        text_extraction_fns=tfidf_text_extraction_fns,
        threshold=tfidf_threshold,
        iterative=False,
        evaluate=False,
    )
    d2v_it = doc2vec_integration(
        base_textbook,
        other_textbooks,
        text_extraction_fn=d2v_text_extraction_fn,
        threshold=d2v_threshold,
        vector_size=d2v_vector_size,
        min_count=d2v_min_count,
        iterative=False,
        evaluate=False,
    )

    for base_section, other_sections in d2v_it.base_to_other_map.items():
        if base_section is None:
            continue
        for other_section in other_sections:
            if (
                tfidf_it.scores[(base_section, other_section)]
                > tfidf_uncertain_threshold
            ):
                tfidf_it.base_to_other_map[base_section].add(other_section)

    tfidf_it.print_matches()
    return tfidf_it.evaluate()
