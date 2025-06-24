#Updated EDA
import os
import statistics
from PyPDF2 import PdfReader

folder_path = "whitby_minutes_2025"

total_pages = 0
total_words = 0
file_count = 0
page_counts = []
word_counts = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                num_pages = len(reader.pages)
                page_counts.append(num_pages)
                total_pages += num_pages

                text = ""
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    except Exception as e:
                        print(f"Skipped page in {filename}: {e}")
                        continue

                words = text.split()
                word_count = len(words)
                word_counts.append(word_count)
                total_words += word_count

                file_count += 1
                print(f"{filename}: {num_pages} pages, {word_count} words")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue


mean_pages = statistics.mean(page_counts) if page_counts else 0
median_pages = statistics.median(page_counts) if page_counts else 0
mean_words = statistics.mean(word_counts) if word_counts else 0
median_words = statistics.median(word_counts) if word_counts else 0

print(f"\nTotal pages: {total_pages} across {file_count} files")
print(f"Mean pages per file: {mean_pages:.2f}")
print(f"Median pages per file: {median_pages}")
print(f"\nTotal words: {total_words}")
print(f"Mean words per file: {mean_words:.2f}")
print(f"Median words per file: {median_words}")


#Yr over yr
import os
import re
from PyPDF2 import PdfReader
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

folder_path = "whitby_minutes_2025"
records = []

#Regex pattern to extract year from filename
year_pattern = re.compile(r"(\d{4})")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        year_match = year_pattern.search(filename)
        if not year_match:
            continue
        year = int(year_match.group(1))
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                word_count = len(text.split())
                page_count = len(reader.pages)
                records.append({
                    "year": year,
                    "filename": filename,
                    "pages": page_count,
                    "words": word_count
                })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

df = pd.DataFrame(records)

#Group by year
summary = df.groupby("year").agg(
    total_files=("filename", "count"),
    total_pages=("pages", "sum"),
    total_words=("words", "sum"),
    mean_pages_per_file=("pages", "mean"),
    mean_words_per_file=("words", "mean"),
    median_pages_per_file=("pages", "median"),
    median_words_per_file=("words", "median")
).reset_index()

print(summary)

#graph
summary.plot.bar(x="year", y="total_files", title="Total Files Per Year", legend=False, color="skyblue")
plt.ylabel("Files")
plt.tight_layout()
plt.show()


#File topic count - under development/incomplete

import os
import re
from PyPDF2 import PdfReader
import pandas as pd

committees = [
    "Accessibility Advisory Committee",
    "Active Transportation and Safe Roads Advisory Committee",
    "Animal Services Appeal Committee",
    "Audit Committee",
    "Brooklin Downtown Development Steering Committee",
    "Committee of Adjustment",
    "Committee of the Whole",
    "Compliance Audit Committee",
    "Downtown Whitby Development Steering Committee",
    "Heritage Whitby Advisory Committee",
    "Joint Accessibility Advisory and Diversity and Inclusion Advisory Committees",
    "Municipal Licensing and Standards Committee",
    "Property Standards Appeal Committee",
    "Public Meetings",
    "Regular Council Meetings",
    "Special Council Meetings",
    "Whitby Diversity and Inclusion Advisory Committee",
    "Whitby Sustainability Advisory Committee"
]

committee_keywords = [c.lower() for c in committees]

folder_path = "whitby_minutes_2025"
data = []

#Process all files for topic
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder_path, filename)
        year_match = re.search(r"\d{4}", filename)
        year = int(year_match.group()) if year_match else None

        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text.lower()

                matched = []
                for i, committee in enumerate(committee_keywords):
                    if committee in text:
                        matched.append(committees[i])

                data.append({
                    "filename": filename,
                    "year": year,
                    "committee": "; ".join(matched) if matched else "Unclassified"
                })

df = pd.DataFrame(data)

#Count files per committee category
category_counts = (
    df["committee"]
    .str.split("; ")
    .explode()
    .value_counts()
    .reset_index()
)
category_counts.columns = ["committee", "file_count"]

print("File classification count by committee:\n")
print(category_counts.to_string(index=False))


#NER
import os
import re
from PyPDF2 import PdfReader
import spacy
import pandas as pd
from collections import Counter

nlp = spacy.load("en_core_web_sm")
folder_path = "whitby_minutes_2025"
ner_data = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    except Exception as e:
                        print(f"Skipped a page in {filename}: {e}")
                        continue

                #Truncate text
                text = re.sub(r"\s+", " ", text.strip())[:30000]

                #Run NER
                try:
                    doc = nlp(text)
                    entities = [ent.label_ for ent in doc.ents]
                    entity_counts = Counter(entities)
                    entity_counts["filename"] = filename
                    ner_data.append(entity_counts)
                except Exception as e:
                    print(f"NER failed on {filename}: {e}")
                    continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

df_ner = pd.DataFrame(ner_data).fillna(0)
df_ner = df_ner.astype({col: int for col in df_ner.columns if col != "filename"})
print(df_ner.head())

#NER graph
import pandas as pd
import matplotlib.pyplot as plt

df = df_ner.copy()
non_entity_columns = ["filename"]
entity_df = df.drop(columns=non_entity_columns)

entity_sums = entity_df.sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
entity_sums.plot(kind="bar", color="skyblue")
plt.title("Total Count of Named Entity Types Across All Files")
plt.xlabel("Entity Type")
plt.ylabel("Total Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis='y')
plt.show()

