# Image Search Engine (Harry Potter Fandom)

This work aims to replicate the functionality of early image search engines like Google Images by indexing textual surrogates associated with images scraped from the [Harry Potter Fandom Wiki](https://harrypotter.fandom.com/wiki/Harry_Potter_Wiki) and allowing users to search for relevant images via a web interface.

## Features

*   **Web Scraping:** Collects image URLs and associated text (page title, captions, alt-text) from Fandom wiki pages (`Scrapper.py`).
*   **Text Preprocessing:** Cleans and processes text surrogates using NLTK (tokenization, lowercasing, stopword removal, stemming) (`preprocessing.py`).
*   **Indexing:** Builds an inverted index, document frequency map, document lengths, and global term statistics from the processed text (`indexing.py`, `run_indexing.py`).
*   **Ranking Models:** Implements three different ranking algorithms:
    *   Vector Space Model (VSM) with TF-IDF (`ranking_models.py`).
    *   Okapi BM25 (`ranking_models.py`).
    *   Language Model with Dirichlet Smoothing (`ranking_models.py`).
*   **Web Interface:** Provides a simple web UI built with Flask to enter search queries, select a ranking model, and view ranked image results (`app.py`, `templates/search.html`).
*   **Persistence:** Saves the built index to disk (`search_index.pkl`) for efficient loading by the web application.

## Technologies Used

*   **Language:** Python 3
*   **Web Framework:** Flask
*   **Web Scraping:** Requests, BeautifulSoup4
*   **Text Processing:** NLTK
*   **Data Persistence:** Pickle
*   **Web Server (for deployment):** Gunicorn (recommended, not included in dev dependencies)
*   **Frontend:** HTML, CSS (with Jinja2 templating)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    # OR (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

There are two main steps: creating the search index (offline) and running the web server (online).

**1. Prepare Data & Build Index (Offline Step)**

*   **(Optional) Run the Scraper:** If you don't have the `fandom_harrypotter_images.csv` file or need to update it, you can run the scraper. **Warning:** Be mindful of Fandom's terms of service and `robots.txt`. Use the built-in delay and do not run excessively.
    ```bash
    python Scrapper.py
    ```
    Ensure the output CSV file (`fandom_harrypotter_images.csv` by default) contains at least 1000 image entries with text surrogates.

*   **Run the Indexing Script:** This reads the CSV, preprocesses the text, builds all necessary index structures, and saves them to `search_index.pkl`.
    ```bash
    python run_indexing.py
    ```
    This step needs to be completed successfully before starting the web server. It may take some time depending on the dataset size.

**2. Run the Web Server (Online Step)**

*   Once `search_index.pkl` exists, start the Flask development server:
    ```bash
    python app.py
    ```
*   The server will likely start on `http://127.0.0.1:5001/` or `http://localhost:5001/`.

## Usage

1.  Open your web browser and navigate to the URL provided when you started the Flask server (e.g., `http://localhost:5001/`).
2.  You will see the search interface.
3.  Enter your query terms (e.g., "Harry Potter wand", "Dumbledore portrait", "Hogwarts castle").
4.  Select the desired ranking model (BM25, VSM, LM) from the dropdown.
5.  Click the "Search" button.
6.  The page will reload displaying the ranked image results based on your query and the chosen model. Images link to their original Fandom wiki page URL.

## Configuration

*   **`Scrapper.py`:** `CONFIG` dictionary at the top controls the start URL, domain restrictions, request delay, max images, etc.
*   **`run_indexing.py`:** Constants `CSV_FILE` and `INDEX_OUTPUT_FILE` define input/output paths.
*   **`app.py`:** Constants `INDEX_FILE` and `RESULTS_PER_PAGE` control the index location and the number of results shown.
*   **`ranking_models.py`:** BM25 parameters (`k1`, `b`) and LM smoothing parameter (`mu`) can be adjusted within the respective functions.

## Deployment Notes

*   For deployment (e.g., PythonAnywhere, Render, Elastic Beanstalk), use a production WSGI server like Gunicorn. Remove `gunicorn` from `requirements.txt` if deploying to Elastic Beanstalk (it provides its own).
*   Ensure the `search_index.pkl` file is deployed alongside the application code.
*   Handle NLTK data downloads appropriately on the server environment (e.g., using `.ebextensions` on EB).
*   Set `debug=False` in `app.run()` or remove the `app.run()` block entirely when using a production server like Gunicorn.

## Disclaimer

Web scraping should be done responsibly. Always check the `robots.txt` file and the Terms of Service of the target website (e.g., Fandom) before running any web crawler. Respect request delays to avoid overloading the site's servers. This scraper is provided for educational purposes as part of the assignment requirements.
