import pdfplumber
import pandas as pd
from pathlib import Path
import re

pdf_dir = Path("/Users/shriramumapathy/Desktop/DocsOnRaG Project/Docs On RAG/python_docs")
target_versions = ["python-3.13-docs", "python-3.14-docs"]

data = {"input": [], "output": [], "version": []}
for version_dir in pdf_dir.iterdir():
    if not version_dir.is_dir() or version_dir.name not in target_versions:
        continue

    print(f"Processing {version_dir.name}")
    pdf_subdir = version_dir / "docs-pdf"
    for pdf_file in pdf_subdir.glob("*.pdf"):
        print(f"  - {pdf_file.name}")
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text(layout=True)
                    if text:
                        full_text += text + "\n\n"

                # Look for TOC-like lines (e.g., "1.1 Title ... 3")
                toc_pattern = r'(?m)^\s*(\d+\.\d*\.?\s+.*?)\s+\.\s+\d+$'
                toc_matches = re.findall(toc_pattern, full_text.split("contents", 1)[-1], re.I)
                print(f"    Found {len(toc_matches)} TOC entries")

                for title in toc_matches:
                    title_clean = re.sub(r'\s+', ' ', title.strip().lower())
                    question = f"What is {title_clean} in Python {version_dir.name.split('-')[1]}?"
                    # Search for title in full text
                    start = full_text.lower().find(title_clean)
                    if start == -1:
                        continue
                    end = full_text.find("\n\n", start + len(title_clean))
                    if end == -1:
                        end = len(full_text)
                    content = full_text[start:end].replace("\n", " ").strip()
                    if len(content) > 50:
                        data["input"].append(question)
                        data["output"].append(content)
                        data["version"].append(version_dir.name.split('-')[1])
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

df = pd.DataFrame(data)
df.to_csv("python_docs_dataset.csv", index=False)
print(f"Dataset created with {len(df)} entries")