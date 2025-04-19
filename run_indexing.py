# run_indexing.py
import csv
import pickle
from indexing import build_inverted_index, build_global_term_counts
from preprocessing import preprocess_text # Make sure preprocess_text is available

# --- Configuration ---
CSV_FILE = 'fandom_harrypotter_images.csv'
INDEX_OUTPUT_FILE = 'search_index.pkl'

# --- Load Data from CSV ---
print(f"Loading data from {CSV_FILE}...")
docs = {}
# Create a mapping from an integer ID to the image data
doc_id_to_data = {}
current_id = 0
try:
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Use a simple integer as doc_id for the index
            doc_id = current_id
            text_surrogate = row.get('text_surrogate', '')
            if text_surrogate: # Only index if there's text
                docs[doc_id] = text_surrogate
                # Store the mapping for later retrieval
                doc_id_to_data[doc_id] = {
                    'image_url': row.get('image_url'),
                    'page_url': row.get('page_url'),
                    'text_surrogate': text_surrogate
                }
                current_id += 1
            else:
                print(f"Skipping row with missing text_surrogate: {row.get('image_url')}")

    print(f"Loaded {len(docs)} documents with text surrogates.")
    if not docs:
        raise ValueError("No documents loaded. Check CSV file and text_surrogate column.")

except FileNotFoundError:
    print(f"Error: CSV file '{CSV_FILE}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()


# --- Build Index ---
print("Building inverted index...")
# build_inverted_index returns: inverted_index, doc_lengths, N
inverted_index, doc_lengths, N = build_inverted_index(docs)
print(f"Indexed {N} documents.")
print(f"Vocabulary size: {len(inverted_index)} terms.")

# --- Build Document Frequency (df) ---
print("Calculating document frequency (df)...")
df = {term: len(posting_list) for term, posting_list in inverted_index.items()}

# --- Build Global Term Counts (for LM) ---
print("Calculating global term counts...")
term_counts, total_tokens = build_global_term_counts(inverted_index)
print(f"Total tokens in collection: {total_tokens}")

# --- Prepare Data for Saving ---
index_data = {
    'inverted_index': inverted_index,
    'doc_lengths': doc_lengths,
    'N': N,
    'df': df,
    'term_counts': term_counts,
    'total_tokens': total_tokens,
    'doc_id_to_data': doc_id_to_data # Include the mapping
}

# --- Save Index Data ---
print(f"Saving index data to {INDEX_OUTPUT_FILE}...")
try:
    with open(INDEX_OUTPUT_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    print("Index saved successfully.")
except Exception as e:
    print(f"Error saving index: {e}")