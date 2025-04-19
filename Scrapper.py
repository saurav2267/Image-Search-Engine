import requests
from bs4 import BeautifulSoup
import csv
import time
import os
from urllib.parse import urljoin, urlparse
from collections import deque # Efficient queue for URLs
from typing import List, Dict, Optional, Any, Set # Make sure Optional and Set are included

# --- Configuration ---
CONFIG = {
    "START_URL": "https://harrypotter.fandom.com/wiki/Harry_Potter_Wiki", # Starting point
    "BASE_URL": "https://harrypotter.fandom.com",
    "ALLOWED_DOMAIN": "harrypotter.fandom.com", # Only crawl within this domain
    "MAX_IMAGES_TO_GATHER": 1050, # Target number of images
    "OUTPUT_CSV_FILE": "fandom_harrypotter_images.csv",
    "REQUEST_DELAY_SECONDS": 2.0, # IMPORTANT: Be polite! >= 2 seconds is safer.
    "REQUEST_TIMEOUT_SECONDS": 15,
    # Basic User-Agent to look less like a default script
    "HEADERS": {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    },
    # Patterns to exclude from crawling (adjust as needed)
    "EXCLUDED_URL_PATTERNS": [
        "/Special:", "/User:", "/User_blog:", "/Template:", "/Talk:",
        "/Category:", "/Help:", "/File:", "/MediaWiki:",
        "action=edit", "action=history", "oldid=",
        "#", ".png", ".jpg", ".jpeg", ".gif", ".svg" # Exclude direct links to images/files in crawl queue
    ]
}

# --- Helper Functions ---

def fetch_page(url: str, headers: dict, timeout: int) -> Optional[str]:
    """Fetches HTML content for a given URL."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        # Check content type to avoid trying to parse non-html
        if 'text/html' in response.headers.get('Content-Type', ''):
            return response.text
        else:
            print(f"Skipping non-HTML content at: {url}")
            return None
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching {url}: {e}")
        return None

def get_absolute_url(base: str, relative: str) -> str:
    """Constructs an absolute URL from base and relative paths."""
    return urljoin(base, relative)

def is_valid_url(url: str, base_url: str, allowed_domain: str, excluded_patterns: list) -> bool:
    """Checks if a URL is valid for crawling."""
    parsed_url = urlparse(url)
    # 1. Check if it's within the allowed domain
    if parsed_url.netloc != allowed_domain:
        return False
    # 2. Check if it's an http/https scheme
    if parsed_url.scheme not in ['http', 'https']:
        return False
    # 3. Check against excluded patterns
    for pattern in excluded_patterns:
        if pattern in url:
            return False
    return True

def extract_image_and_text(soup: BeautifulSoup, page_url: str, base_url: str) -> List[Dict[str, str]]:
    """
    Extracts images and attempts to find associated text.
    ** This function likely needs SIGNIFICANT adjustments based on Fandom's HTML **
    """
    extracted_data = []
    page_title = soup.title.string if soup.title else ""

    # --- Strategy 1: Look for images within figures with figcaptions ---
    # Inspect Fandom pages to find the correct selectors! These are guesses.
    figures = soup.find_all('figure', class_=lambda x: x and 'image' in x) # Example selector
    for figure in figures:
        img_tag = figure.find('img')
        caption_tag = figure.find('figcaption')
        if img_tag and img_tag.get('src'):
            img_url = get_absolute_url(base_url, img_tag['src'])
            # Attempt to get higher-res source if available (common in wikis)
            if img_tag.get('data-src'):
                 img_url = get_absolute_url(base_url, img_tag['data-src'])

            text_parts = [page_title]
            if caption_tag and caption_tag.get_text(strip=True):
                text_parts.append(caption_tag.get_text(strip=True))
            elif img_tag.get('alt'): # Fallback to alt text
                 text_parts.append(img_tag.get('alt', ''))

            # Add text from nearby paragraphs (more complex, optional)
            # prev_p = figure.find_previous_sibling('p')
            # if prev_p: text_parts.append(prev_p.get_text(strip=True))

            surrogate = ' '.join(filter(None, text_parts))
            extracted_data.append({
                'page_url': page_url,
                'image_url': img_url,
                'text_surrogate': ' '.join(surrogate.split()) # Clean whitespace
            })

    # --- Strategy 2: Look for images in common wiki "thumb" divs (alternative) ---
    # Example: find divs with class 'thumb' or 'image-thumbnail' etc.
    # thumbs = soup.select('div.thumb, div.image-thumbnail') # Adjust selector
    # for thumb in thumbs:
    #     # ... similar logic to extract img src, captions, alt text ...
    #     pass

    # --- Strategy 3: General images (less reliable context) ---
    # all_imgs = soup.find_all('img')
    # for img_tag in all_imgs:
         # if img_tag.get('src'):
             # img_url = get_absolute_url(base_url, img_tag['src'])
             # # Check if already processed by better methods?
             # # Try to find context based on parent elements / alt text only
             # # ... this gets less precise ...

    # Deduplicate based on image URL before returning
    seen_image_urls = set()
    final_data = []
    for item in extracted_data:
        if item['image_url'] not in seen_image_urls:
            final_data.append(item)
            seen_image_urls.add(item['image_url'])

    return final_data


def find_internal_links(soup: BeautifulSoup, base_url: str, allowed_domain: str, excluded_patterns: list) -> set:
    """Finds valid internal links on the page to crawl next."""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_url = get_absolute_url(base_url, href)
        if is_valid_url(absolute_url, base_url, allowed_domain, excluded_patterns):
            links.add(absolute_url)
    return links

def save_data_to_csv(data: List[Dict[str, str]], filename: str):
    """Saves the collected data to a CSV file."""
    if not data:
        print("No data collected to save.")
        return
    print(f"Saving data for {len(data)} images to {filename}...")
    try:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['page_url', 'image_url', 'text_surrogate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully saved data to {filename}")
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")


# --- Main Crawler Logic ---
def run_fandom_crawler(config: dict):
    """Main orchestration function."""
    print("--- Starting Fandom Crawler ---")
    print(f"WARNING: Scraping wikis like Fandom can be complex and may be against Terms of Service.")
    print(f"Ensure politeness delay ({config['REQUEST_DELAY_SECONDS']}s) is respected.")
    print(f"Check {config['BASE_URL']}/robots.txt")

    urls_to_visit = deque([config["START_URL"]])
    visited_urls = set()
    collected_images_data = []
    img_count = 0

    while urls_to_visit and img_count < config["MAX_IMAGES_TO_GATHER"]:
        current_url = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        if not is_valid_url(current_url, config["BASE_URL"], config["ALLOWED_DOMAIN"], config["EXCLUDED_URL_PATTERNS"]):
            continue

        print(f"\nVisiting: {current_url}")
        visited_urls.add(current_url)

        html_content = fetch_page(current_url, config["HEADERS"], config["REQUEST_TIMEOUT_SECONDS"])

        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract images and text from this page
            page_image_data = extract_image_and_text(soup, current_url, config["BASE_URL"])
            print(f"Found {len(page_image_data)} potential images on page.")

            # Add new images if space allows
            for img_data in page_image_data:
                 if img_count < config["MAX_IMAGES_TO_GATHER"]:
                     # Basic filter: avoid tiny images or known junk (adjust size check)
                     # This requires analyzing image URLs or dimensions, which adds complexity
                     # Simple filter example: Avoid generic Fandom assets
                     if 'static.wikia.nocookie.net/common/skins/common/images' not in img_data['image_url']:
                         collected_images_data.append(img_data)
                         img_count += 1
                 else:
                     break # Stop adding images if max reached

            print(f"Total images collected: {img_count}/{config['MAX_IMAGES_TO_GATHER']}")

            # Find new links to visit
            if img_count < config["MAX_IMAGES_TO_GATHER"]:
                new_links = find_internal_links(soup, config["BASE_URL"], config["ALLOWED_DOMAIN"], config["EXCLUDED_URL_PATTERNS"])
                for link in new_links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                print(f"Queue size: {len(urls_to_visit)}")


        # POLITE DELAY - crucial for not getting blocked
        print(f"Waiting {config['REQUEST_DELAY_SECONDS']} seconds...")
        time.sleep(config["REQUEST_DELAY_SECONDS"])


    print(f"\n--- Crawler Finished ---")
    print(f"Visited {len(visited_urls)} pages.")
    print(f"Collected data for {len(collected_images_data)} images.")

    # Save the data
    save_data_to_csv(collected_images_data, config["OUTPUT_CSV_FILE"])


# --- Main Execution ---
if __name__ == "__main__":
    # !!! Before running:
    # 1. Manually check https://harrypotter.fandom.com/robots.txt
    # 2. Inspect image elements (e.g., in character pages) using browser developer tools.
    # 3. Adjust the selectors in `extract_image_and_text` function based on inspection.
    # 4. Consider if the delay is sufficient. Increase if needed.
    run_fandom_crawler(CONFIG)