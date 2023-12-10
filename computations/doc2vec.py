import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from textbooks.data import initialise_textbooks
from textbooks.integration import SimilarityBasedTextbookIntegration

nltk.download("punkt")


def preprocess(doc):
    return word_tokenize(doc.lower())


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def doc2vec_vector_computation(
    corpus, text_extraction_fn, vector_size=100, min_count=50, epochs=40
):
    preprocessed_corpus = {
        section: preprocess(text_extraction_fn(section)) for section in corpus
    }

    train_corpus = [
        TaggedDocument(words=text, tags=[(section.textbook, section)])
        for section, text in preprocessed_corpus.items()
    ]

    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    return {
        section: model.infer_vector(text)
        for section, text in preprocessed_corpus.items()
    }


def doc2vec_integration(text_extraction_fn, similarity_threshold):
    base_textbook, other_textbooks = initialise_textbooks()
    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        scoring_fn=cos,
        threshold=similarity_threshold,
    )

    vectors = doc2vec_vector_computation(integrated_textbook.corpus, text_extraction_fn)

    # Integrate the sections with the computed vectors
    integrated_textbook.add_section_vectors(vectors)
    integrated_textbook.integrate_sections()
    integrated_textbook.print_matches()
