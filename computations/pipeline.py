from computations.doc2vec import doc2vec_integration
from computations.tfidf import tfidf_cosine_similarity_integration


def pipeline_integration(base_textbook, other_textbooks, models):
    integrated_textbooks = []
    uncertain_thresholds = []
    for model in models:
        integrated_textbook = model["fn"](
            base_textbook=base_textbook,
            other_textbooks=other_textbooks,
            evaluate=False,
            **model["kwargs"],
        )
        integrated_textbooks.append(integrated_textbook)
        if "uncertain_threshold" in model:
            uncertain_thresholds.append(model["uncertain_threshold"])

    for i in range(len(integrated_textbooks) - 1, 0, -1):
        earlier, later = integrated_textbooks[i - 1 : i + 1]
        lower, upper = uncertain_thresholds[i - 1]
        for base_section, other_sections in later.base_to_other_map.items():
            if base_section is not None:
                earlier.base_to_other_map[base_section] |= {
                    other_section
                    for other_section in other_sections
                    if lower < earlier.scores[(base_section, other_section)] < upper
                }

    earlier.print_matches()
    return earlier.evaluate()


def tfidf_doc2vec_pipeline(
    base_textbook,
    other_textbooks,
    tfidf_text_extraction_fns,
    tfidf_threshold,
    tfidf_uncertain_threshold,
    doc2vec_text_extraction_fn,
    doc2vec_threshold,
):
    if tfidf_uncertain_threshold[1] > tfidf_threshold:
        raise ValueError("Uncertain threshold cannot overlap main threshold")
    return pipeline_integration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        models=[
            {
                "fn": tfidf_cosine_similarity_integration,
                "kwargs": {
                    "text_extraction_fns": tfidf_text_extraction_fns,
                    "threshold": tfidf_threshold,
                },
                "uncertain_threshold": tfidf_uncertain_threshold,
            },
            {
                "fn": doc2vec_integration,
                "kwargs": {
                    "text_extraction_fn": doc2vec_text_extraction_fn,
                    "threshold": doc2vec_threshold,
                },
            },
        ],
    )
