import math
from collections import Counter
from preprocessing import preprocess_text

def rank_vsm(query_text, inverted_index, df, N, doc_lengths):
    """
    Vector Space Model with TF-IDF scoring and normalization.
    Returns list of (doc_id, score) sorted descending by score.
    """
    query_tokens = preprocess_text(query_text)
    query_tf = Counter(query_tokens)
    scores = {}
    
    # Calculate query vector length for normalization
    query_length = 0
    for term, qf in query_tf.items():
        if term in inverted_index:
            idf = math.log2((N / (df[term] + 1)))
            query_length += (qf * idf) ** 2
    query_length = math.sqrt(query_length)

    # Calculate document scores
    for term, qf in query_tf.items():
        if term not in inverted_index:
            continue

        # IDF with smoothing
        idf = math.log2((N / (df[term] + 1)))
        w_qt = qf * idf

        # Accumulate scores for docs that have this term
        for doc_id, tf_dt in inverted_index[term].items():
            # TF with length normalization
            doc_len = doc_lengths[doc_id]
            w_dt = (tf_dt / doc_len) * idf
            scores[doc_id] = scores.get(doc_id, 0) + (w_dt * w_qt)

    # Normalize scores by query length
    for doc_id in scores:
        scores[doc_id] /= query_length

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def rank_bm25(query_text, inverted_index, df, N, doc_lengths, k1=1.5, b=0.75):
    """
    BM25 ranking with optimized parameters.
    Returns list of (doc_id, score) sorted descending by BM25 score.
    """
    query_tokens = preprocess_text(query_text)
    query_tf = Counter(query_tokens)

    # Calculate average document length
    avgdl = sum(doc_lengths.values()) / float(N)
    scores = {}

    for term, qf in query_tf.items():
        if term not in inverted_index:
            continue

        df_t = df[term]
        # Enhanced IDF with smoothing
        idf = math.log2((N - df_t + 0.5) / (df_t + 0.5) + 1)

        posting_list = inverted_index[term]
        for doc_id, f_dt in posting_list.items():
            doc_len = doc_lengths[doc_id]
            # Enhanced BM25 formula with query term frequency
            numerator = f_dt * (k1 + 1) * qf
            denominator = f_dt + k1 * (1 - b + b * (doc_len / avgdl))
            score = idf * (numerator / denominator)
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def rank_language_model(query_text, inverted_index, df, doc_lengths,
                       term_counts, total_tokens, mu=2500):
    """
    LM Model with Dirichlet smoothing and mu parameter of 2500.
    For each document, sum log( P(t|d) ) for t in query.
    Returns list of (doc_id, score) in descending order (score = log-prob).
    """
    query_tokens = preprocess_text(query_text)
    query_tf = Counter(query_tokens)
    scores = {}

    for doc_id, doc_len in doc_lengths.items():
        log_prob_sum = 0.0

        for term, qf in query_tf.items():
            f_dt = 0
            if term in inverted_index and doc_id in inverted_index[term]:
                f_dt = inverted_index[term][doc_id]
            c_t = term_counts.get(term, 0)

            # Enhanced Dirichlet smoothing with query term frequency
            p_t_d = (f_dt + mu * (c_t / total_tokens)) / (doc_len + mu)
            if p_t_d > 0:
                log_prob_sum += qf * math.log(p_t_d)

        scores[doc_id] = log_prob_sum

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs 