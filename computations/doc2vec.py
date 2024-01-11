import nltk
import numpy as np
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from textbooks.integration import SimilarityBasedTextbookIntegration

nltk.download("punkt", quiet=True)


def build_doc2vec_model(tagged_corpus, vector_size, min_count, epochs):
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(tagged_corpus)
    model.train(tagged_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def preprocess_corpus(sections, text_extraction_fn):
    return {
        section: word_tokenize(text_extraction_fn(section).lower())
        for section in sections
    }


def tag_corpus(preprocessed_corpus):
    return [
        TaggedDocument(words=text, tags=[(section.textbook, section)])
        for section, text in preprocessed_corpus.items()
    ]


def doc2vec_integration(
    base_textbook,
    other_textbooks,
    text_extraction_fn,
    iterative,
    threshold,
    vector_size,
    min_count,
    epochs,
    evaluate=True,
):
    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        scoring_fn=lambda a, b: np.dot(a, b) / (norm(a) * norm(b)),
        threshold=threshold,
        iterative=iterative,
    )

    preprocessed_corpus = preprocess_corpus(
        integrated_textbook.corpus, text_extraction_fn
    )
    tagged_corpus = tag_corpus(preprocessed_corpus)
    model = build_doc2vec_model(tagged_corpus, vector_size, min_count, epochs)
    vectors = {
        section: model.infer_vector(text)
        for section, text in preprocessed_corpus.items()
    }
    integrated_textbook.add_section_vectors(vectors)

    if iterative:
        for match in integrated_textbook.integrate_sections():
            if match is None:
                continue
            base_section, other_section = match
            base_section.extend(other_section)

            new_preprocessed_corpus = preprocess_corpus(
                [base_section], text_extraction_fn
            )
            new_tagged_corpus = tag_corpus(new_preprocessed_corpus)

            model.build_vocab(new_tagged_corpus, update=True)
            model.train(
                new_tagged_corpus,
                total_examples=model.corpus_count,
                epochs=model.epochs,
            )
            new_vectors = {
                section: model.infer_vector(text)
                for section, text in new_preprocessed_corpus.items()
            }
            integrated_textbook.add_section_vectors(new_vectors)
    else:
        integrated_textbook.integrate_sections()

    if not evaluate:
        return integrated_textbook
    integrated_textbook.print_matches()
    return integrated_textbook.evaluate()
