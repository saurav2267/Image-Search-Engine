import os
import pandas as pd
from indexing import build_inverted_index, build_global_term_counts
from ranking_models import rank_vsm, rank_bm25, rank_language_model
from preprocessing import preprocess_text

def main():
    print("\n--- Starting Image Search Engine ---")

    # 1. Load image CSV dataset
    csv_path = "fandom_harrypotter_images.csv"
    df = pd.read_csv(csv_path)
    doc_map = {row['image_url']: row['text_surrogate'] for _, row in df.iterrows()}
    print(f"Loaded {len(doc_map)} image documents.")

    # 2. Build inverted index
    inverted_index, doc_lengths, N = build_inverted_index(doc_map)
    df_stats = {term: len(postings) for term, postings in inverted_index.items()}
    term_counts, total_tokens = build_global_term_counts(inverted_index)
    print(f"Inverted index built with {len(inverted_index)} terms.")

    # 3. Sample query input (manual for now)
    while True:
        query = input("\nEnter a search query (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            break

        print("\nTop 5 Results (BM25):")
        results = rank_bm25(query, inverted_index, df_stats, N, doc_lengths)
        for rank, (doc_id, score) in enumerate(results[:5], start=1):
            print(f"{rank}. {doc_id}")
            print(f"   → Caption: {doc_map[doc_id]}\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
