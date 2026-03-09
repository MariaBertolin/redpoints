import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_reference_matrix(reference_texts, vectorizer):
    """
    Vectorize reference listings once.
    """
    if reference_texts is None or len(reference_texts) == 0:
        raise ValueError("reference_texts cannot be empty")

    reference_texts = list(reference_texts)
    reference_vectors = vectorizer.transform(reference_texts)

    return reference_texts, reference_vectors


def find_similar_listings(query_text, reference_texts, reference_vectors, vectorizer, k=5):
    """
    Find the k most similar reference listings to one query text.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")

    if reference_texts is None or len(reference_texts) == 0:
        raise ValueError("reference_texts cannot be empty")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    reference_texts = list(reference_texts)
    k = min(k, len(reference_texts))

    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, reference_vectors)[0]

    top_k_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_k_indices:
        results.append({
            "index": int(idx),
            "text": reference_texts[idx],
            "similarity": float(similarities[idx])
        })

    return {
        "query": query_text,
        "similar": results
    }


def batch_find_similar_listings(query_texts, reference_texts, reference_vectors, vectorizer, k=5):
    """
    Find similar listings for multiple query texts.
    """
    results = []
    for query_text in query_texts:
        results.append(
            find_similar_listings(query_text, reference_texts, reference_vectors, vectorizer, k)
        )
    return results


def similarity_stats(query_text, reference_vectors, vectorizer):
    """
    Compute similarity statistics for one query.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")

    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, reference_vectors)[0]

    return {
        "query": query_text,
        "mean_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities))
    }