from top2vec import Top2Vec

from textbooks.integration import QueryBasedTextbookIntegration


def top2vec_integration(
    base_textbook, other_textbooks, text_extraction_fn, similarity_threshold
):
    corpus = base_textbook.all_subsections()
    preprocessed_corpus = [text_extraction_fn(section) for section in corpus]

    model = Top2Vec(documents=preprocessed_corpus, speed="fast-learn", workers=8)

    def top2vec_similarity_function(section):
        doc_scores, doc_ids = model.query_documents(
            query=text_extraction_fn(section), num_docs=1, return_documents=False
        )
        return {"section": corpus[doc_ids[0]], "score": doc_scores[0]}

    integrated_textbook = QueryBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        scoring_fn=top2vec_similarity_function,
        threshold=similarity_threshold,
    )

    integrated_textbook.integrate_sections()
    integrated_textbook.print_matches()
    integrated_textbook.evaluate()
