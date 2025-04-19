# app.py
import nltk
import os
import pickle
from flask import Flask, render_template, request
from ranking_models import rank_vsm, rank_bm25, rank_language_model
# NOTE: preprocessing.py is implicitly used by ranking_models, ensure it's present

# --- Ensure 'punkt' is downloaded into a local folder ---
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# --- Configuration ---
INDEX_FILE = 'search_index.pkl'
RESULTS_PER_PAGE = 20  # Number of images to show

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Index Data (Do this ONCE at startup) ---
print(f"Loading index data from {INDEX_FILE}...")
try:
    with open(INDEX_FILE, 'rb') as f:
        index_data = pickle.load(f)
    inverted_index = index_data['inverted_index']
    doc_lengths = index_data['doc_lengths']
    N = index_data['N']
    df = index_data['df']
    term_counts = index_data['term_counts']
    total_tokens = index_data['total_tokens']
    doc_id_to_data = index_data['doc_id_to_data']  # Load the mapping
    print(f"Index loaded successfully. {N} documents.")
except FileNotFoundError:
    print(f"FATAL ERROR: Index file '{INDEX_FILE}' not found.")
    print("Please run 'run_indexing.py' first.")
    exit()
except Exception as e:
    print(f"FATAL ERROR: Could not load index file: {e}")
    exit()

# --- Web Routes ---

@app.route('/')
def home():
    """Renders the initial search page."""
    return render_template('search.html', query='', results=[], model='bm25', error=None)

@app.route('/search')
def search_results():
    """Handles the search query and displays results on the same template."""
    query = request.args.get('query', '')
    model_choice = request.args.get('model', 'bm25')  # Default to BM25

    results_for_template = []
    error_message = None

    if not query:
        return render_template('search.html', query=query, results=[], model=model_choice, error=None)

    print(f"Received query: '{query}' using model: {model_choice}")

    ranked_docs = []
    try:
        if model_choice == 'vsm':
            ranked_docs = rank_vsm(query, inverted_index, df, N, doc_lengths)
        elif model_choice == 'lm':
            if term_counts is None or total_tokens is None:
                raise ValueError("Language Model components not found in index file.")
            ranked_docs = rank_language_model(query, inverted_index, df, doc_lengths, term_counts, total_tokens)
        else:  # Default to BM25
            ranked_docs = rank_bm25(query, inverted_index, df, N, doc_lengths)

        for doc_id, score in ranked_docs[:RESULTS_PER_PAGE]:
            if doc_id in doc_id_to_data:
                image_info = doc_id_to_data[doc_id]
                results_for_template.append({
                    'doc_id': doc_id,
                    'score': round(score, 4),
                    'image_url': image_info.get('image_url', '#'),
                    'page_url': image_info.get('page_url', '#'),
                    'text_surrogate': image_info.get('text_surrogate', 'No text available')
                })
            else:
                print(f"Warning: doc_id {doc_id} from ranking not found in doc_id_to_data mapping.")

    except Exception as e:
        print(f"Error during ranking: {e}")
        error_message = f"Search failed: {e}"

    return render_template('search.html',
                           query=query,
                           results=results_for_template,
                           model=model_choice,
                           error=error_message)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
