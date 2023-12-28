from functools import partial

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

from textbooks.integration import SimilarityBasedTextbookIntegration

from .tfidf import sklearn_cos, tfidf_vector_computation


def clustering(integrated_textbook, category_extraction_fn, n_clusters):
    section_categories = {
        section: category_extraction_fn(section)
        for section in integrated_textbook.corpus
    }

    # One-Hot Encoding
    category_sets = list(section_categories.values())
    mlb = MultiLabelBinarizer()
    one_hot_encoding = mlb.fit_transform(category_sets)

    # Clustering
    km_model = KMeans(n_clusters=n_clusters)
    clusters = km_model.fit_predict(one_hot_encoding)

    return dict(zip(section_categories, clusters))


def clustering_integration(
    base_textbook,
    other_textbooks,
    clustering_fns,
    text_extraction_fns,
    similarity_threshold,
    weights,
):
    if len(clustering_fns) + len(text_extraction_fns) != len(weights):
        raise ValueError

    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        threshold=similarity_threshold,
    )
    corpus = integrated_textbook.corpus

    tfidf_weights = weights[len(clustering_fns) :]
    tfidf_total_weight = sum(tfidf_weights)
    tfidf_section_vectors = tfidf_vector_computation(
        corpus, text_extraction_fns, tfidf_weights
    )

    cluster_weights = weights[: len(clustering_fns)]
    cluster_dicts = [
        clustering_fn(integrated_textbook) for clustering_fn in clustering_fns
    ]

    def ensemble_similarity_fn(section, other_section):
        section_tfidf_vector = tfidf_section_vectors[section]
        other_section_tfidf_vector = tfidf_section_vectors[other_section]
        tfidf_cosine_similarity = sklearn_cos(
            section_tfidf_vector, other_section_tfidf_vector
        )[0][0]

        cluster_similarity = [
            cluster_dict[section] == cluster_dict[other_section]
            for cluster_dict in cluster_dicts
        ]

        return np.average(
            [tfidf_cosine_similarity] + cluster_similarity,
            weights=[tfidf_total_weight] + cluster_weights,
        )

    integrated_textbook.scoring_fn = ensemble_similarity_fn

    integrated_textbook.integrate_sections()
    integrated_textbook.print_matches()
    integrated_textbook.evaluate()


concept_subject_clustering = partial(
    clustering,
    category_extraction_fn=lambda section: [
        c["subject"] for c in section.concepts.values() if "subject" in c
    ],
    n_clusters=100,
)
concept_name_clustering = partial(
    clustering,
    category_extraction_fn=lambda section: " ".join(
        c["name"] for c in section.concepts.values()
    ),
    n_clusters=100,
)
