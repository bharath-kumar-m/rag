from requests_html import HTMLSession
import pandas as pd
import os
from tqdm import tqdm

CSV_FILE = "bundles.csv"
OUTPUT_DIR = "scraped_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = HTMLSession()

df = pd.read_csv(CSV_FILE)

for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row["advisories"]
    try:
        r = session.get(url)
        r.html.render(timeout=30)  # renders JavaScript

        text = r.html.text
        safe_filename = f"page_{i}.txt"
        with open(os.path.join(OUTPUT_DIR, safe_filename), "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e:
        print(f"❌ Failed {url}: {e}")