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
    km_model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=2024)
    clusters = km_model.fit_predict(one_hot_encoding)

    return dict(zip(section_categories, clusters))


def clustering_integration(
    base_textbook,
    other_textbooks,
    category_extraction_fns,
    n_clusters_options,
    threshold=0.99,
    weights=None,
):
    if weights is None:
        weights = [1] * len(category_extraction_fns)
    if len(n_clusters_options) != len(category_extraction_fns):
        raise ValueError

    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        threshold=threshold,
    )

    cluster_dicts = [
        clustering(
            integrated_textbook,
            category_extraction_fn=category_extraction_fn,
            n_clusters=n_clusters,
        )
        for category_extraction_fn, n_clusters in zip(
            category_extraction_fns, n_clusters_options
        )
    ]

    def ensemble_similarity_fn(section, other_section):
        cluster_similarity = [
            cluster_dict[section] == cluster_dict[other_section]
            for cluster_dict in cluster_dicts
        ]

        return np.average(cluster_similarity, weights=weights)

    integrated_textbook.scoring_fn = ensemble_similarity_fn

    integrated_textbook.integrate_sections()
    integrated_textbook.print_matches()
    return integrated_textbook.evaluate()


def tfidf_clustering_ensemble_integration(
    base_textbook,
    other_textbooks,
    category_extraction_fns,
    n_clusters_options,
    text_extraction_fns,
    threshold,
    weights,
    evaluate=True,
):
    if len(category_extraction_fns) + len(text_extraction_fns) != len(weights):
        raise ValueError
    if len(n_clusters_options) != len(category_extraction_fns):
        raise ValueError

    integrated_textbook = SimilarityBasedTextbookIntegration(
        base_textbook=base_textbook,
        other_textbooks=other_textbooks,
        threshold=threshold,
    )
    corpus = integrated_textbook.corpus

    tfidf_weights = weights[len(category_extraction_fns) :]
    tfidf_total_weight = sum(tfidf_weights)
    tfidf_section_vectors = tfidf_vector_computation(
        corpus, text_extraction_fns, tfidf_weights
    )

    cluster_weights = weights[: len(category_extraction_fns)]
    cluster_dicts = [
        clustering(
            integrated_textbook,
            category_extraction_fn=category_extraction_fn,
            n_clusters=n_clusters,
        )
        for category_extraction_fn, n_clusters in zip(
            category_extraction_fns, n_clusters_options
        )
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
    if not evaluate:
        return integrated_textbook
    integrated_textbook.print_matches()
    return integrated_textbook.evaluate()
