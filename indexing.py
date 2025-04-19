from collections import defaultdict
from preprocessing import preprocess_text

def _create_int_defaultdict():
    """Helper function to create a defaultdict(int), needed for pickling."""
    return defaultdict(int)

def build_inverted_index(docs):
    # Given {doc_id: raw_text}, build an inverted index & doc length map.

    inverted_index = defaultdict(_create_int_defaultdict)
    doc_lengths = {}

    for doc_id, text in docs.items():
        tokens = preprocess_text(text)
        doc_lengths[doc_id] = len(tokens)
        for term in tokens:
            inverted_index[term][doc_id] += 1

    return inverted_index, doc_lengths, len(docs)

def build_global_term_counts(inverted_index):
    """
    For Language Model:
    Returns a dict of {term: total_count_in_collection},plus the total number of tokens in the entire collection.
    """
    term_counts = {}
    total_tokens = 0

    for term, posting_list in inverted_index.items():
        term_sum = sum(posting_list.values())
        term_counts[term] = term_sum
        total_tokens += term_sum

    return term_counts, total_tokens 