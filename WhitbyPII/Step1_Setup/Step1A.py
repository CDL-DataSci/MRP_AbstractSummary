#Step 1A: Data Extraction

import os
import re
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
from ftfy import fix_text
import csv

pdf_directory = "./whitby_minutes_2025" 
output_csv = "step1a.csv"

#Extract text from PDFs
def clean_files(text):
    #remove special characters
    broken_patterns = [
        r"‚Ä¢", r"â€“", r"â€”", r"â€œ", r"â€", r"â€™", r"ÔÇ∑", r"ÔÇö", r"Ã¢", r"â€¦", r"Ã©"
    ]
    for pattern in broken_patterns:
        text = re.sub(pattern, " ", text)
    return text

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        raw_text = "\n".join([page.get_text() for page in doc])
        cleaned_text = " ".join(raw_text.splitlines())
        fixed_text = fix_text(cleaned_text) 
        no_junk = clean_files(fixed_text)
        return no_junk.strip()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

#Extact information from file name
def parse_filename(filename):
    # Remove extension
    name = filename.rsplit('.', 1)[0]
    
    # Remove trailing (1), (2), etc.
    name = re.sub(r'\(\d+\)$', '', name).strip()

    # Split on first dash
    if '-' not in name:
        return filename, None, None 
    category, date_str = name.split('-', 1)
    category = category.strip()
    date_str = date_str.strip()

    # Parse date if possible
    try:
        parsed_date = datetime.strptime(date_str, "%d %b %Y").date()
    except ValueError:
        parsed_date = None

    return category, parsed_date, name 

def batch_extract_text(pdf_dir):
    records = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, fname)
            text = extract_text_from_pdf(full_path)
            category, parsed_date, _ = parse_filename(fname)
            records.append({
                "filename": fname,
                "category": category,
                "date": parsed_date,
                "text": text
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    df = batch_extract_text(pdf_directory)
    df.to_csv("step1a.csv", index=False, quoting=csv.QUOTE_ALL)
