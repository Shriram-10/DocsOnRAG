import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pdfplumber
import re
import nltk
import ssl
from uuid import uuid4
from typing import List, Dict, Optional
import json
from time import time

# Fix NLTK SSL issue
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context
nltk.download('punkt')

def extract_pdf_metadata(pdf: pdfplumber.PDF) -> Dict[str, Optional[str]]:
    """Extract title, version, and doc_type from the first page."""
    start_time = time()
    metadata = {'title': None, 'version': None, 'doc_type': None}
    if pdf.pages:
        text = pdf.pages[0].extract_text()
        if text:
            title_match = re.search(r'^(The Python .+?)(?:\nRelease (\d+\.\d+\.\d+))?', text, re.M)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
                metadata['version'] = title_match.group(2) if title_match.group(2) else 'Unknown'
                if 'Library Reference' in metadata['title']:
                    metadata['doc_type'] = 'Library Reference'
                elif 'Language Reference' in metadata['title']:
                    metadata['doc_type'] = 'Language Reference'
                elif 'What’s New' in metadata['title']:
                    metadata['doc_type'] = 'What’s New'
                else:
                    metadata['doc_type'] = 'Unknown'
    print(f"Metadata extracted in {time() - start_time:.2f}s: {metadata}")
    return metadata

def clean_page_content(page: pdfplumber.page.Page, metadata: Dict[str, Optional[str]]) -> str:
    """Clean headers/footers, preserving content."""
    title = metadata.get('title', 'The Python')
    version = metadata.get('version', r'\d+\.\d+\.\d+')
    header_patterns = [
        re.compile(rf'^{re.escape(title)}(?:, Release {version})?$'),
        re.compile(r'^The Python (?:Library|Language) Reference(?:, Release \d+\.\d+\.\d+)?$'),
        re.compile(r'^Python Documentation$'),
    ]
    footer_patterns = [
        re.compile(r'^Chapter \w+( \w+)?$'),
        re.compile(r'^\d+$'),
        re.compile(r'^[ivx]+$', re.I),
        re.compile(r'^CONTENTS$', re.I),
        re.compile(r'^\d+\s*$'),
        re.compile(rf'^{re.escape(title)}\s*$'),
    ]
    
    lines = page.extract_text_lines()
    cleaned_lines = []
    top_margin = page.height * 0.1
    bottom_margin = page.height * 0.9
    
    for line in lines:
        text = line['text'].strip()
        if not text:
            continue
        is_header_footer = (line['top'] < top_margin or line['bottom'] > bottom_margin or
                            any(p.match(text) for p in header_patterns + footer_patterns))
        if not is_header_footer:
            cleaned_lines.append(text)
    
    content = '\n'.join(cleaned_lines)
    print(f"Page {page.page_number} cleaned content: {content[:100]}..." if content else f"Page {page.page_number} empty")
    return content

def cache_pages(pdf: pdfplumber.PDF, metadata: Dict[str, Optional[str]]) -> Dict[int, str]:
    """Cache cleaned content for all pages."""
    start_time = time()
    cached_pages = {}
    for page in pdf.pages:
        cached_pages[page.page_number] = clean_page_content(page, metadata)
    print(f"Cached {len(cached_pages)} pages in {time() - start_time:.2f}s")
    return cached_pages

def detect_index_pages(pdf: pdfplumber.PDF) -> List[tuple]:
    """Detect table of contents pages."""
    start_time = time()
    index_pages = []
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text and (re.search(r'^(CONTENTS|Table of Contents)\b', text, re.I) or 
                     re.match(r'^\d+(?:\.\d+)*\s', text)):
            index_pages.append((page_num, text))
            print(f"Found TOC on page {page_num}: {text[:50]}...")
    print(f"TOC detection completed in {time() - start_time:.2f}s, found {len(index_pages)} pages out of {len(pdf.pages)}")
    return index_pages

def parse_index(index_pages: List[tuple]) -> List[Dict]:
    """Parse TOC into section entries for all levels (X, X.Y, X.Y.Z, etc.)."""
    start_time = time()
    index_entries = []
    pattern = r'^(\d+(?:\.\d+)*)\s+(.+?)\s+\.*\s+(\d+)$'
    for page_num, text in index_pages:
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('CONTENTS'):
                continue
            match = re.match(pattern, line)
            if match:
                section_num = match.group(1)
                title = match.group(2).strip()
                page_num = int(match.group(3))
                index_entries.append({
                    'section_number': section_num,
                    'title': title,
                    'page_number': page_num
                })
                print(f"Parsed TOC entry: {section_num} - {title} (page {page_num})")
            else:
                print(f"Skipped TOC line (page {page_num}): '{line}'")
    if not index_entries:
        print("No TOC entries parsed!")
    print(f"TOC parsing completed in {time() - start_time:.2f}s, parsed {len(index_entries)} entries")
    return sorted(index_entries, key=lambda x: x['section_number'])

def build_hierarchy(index_entries: List[Dict]) -> Dict:
    """Build section hierarchy, assigning page numbers to parents from first child."""
    start_time = time()
    hierarchy = {}
    for entry in index_entries:
        section_num = entry['section_number']
        levels = section_num.split('.')
        current = hierarchy
        for i, level in enumerate(levels):
            level_key = '.'.join(levels[:i+1])
            if level_key not in current:
                current[level_key] = {
                    'title': None,
                    'page_number': None,
                    'children': {}
                }
            if i == len(levels) - 1:
                current[level_key]['title'] = entry['title']
                current[level_key]['page_number'] = entry['page_number']
            if i > 0 and current[level_key]['page_number'] is not None:
                parent_key = '.'.join(levels[:i])
                parent = hierarchy
                for j in range(i):
                    parent = parent[levels[j]]
                if parent[parent_key]['page_number'] is None:
                    parent[parent_key]['page_number'] = current[level_key]['page_number']
            current = current[level_key]['children']
    print(f"Hierarchy built in {time() - start_time:.2f}s with {len(hierarchy)} top-level sections")
    return hierarchy

def segment_content(text: str, section_num: str, section_title: str, start_page: int, end_page: int, 
                    pdf_name: str, version: str, doc_type: str) -> List[Dict]:
    """Segment content into text/code chunks."""
    start_time = time()
    chunks = []
    current_chunk = []
    content_type = 'text'
    is_code = False

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>>>'):
            if current_chunk:
                chunks.append({
                    'id': str(uuid4()),
                    'pdf_name': pdf_name,
                    'version': version,
                    'doc_type': doc_type,
                    'section_number': section_num,
                    'section_path': section_num.split('.'),
                    'section_titles': [f"Section {'.'.join(section_num.split('.')[:i+1])}" 
                                    for i in range(len(section_num.split('.')) - 1)] + [section_title],
                    'content': '\n'.join(current_chunk),
                    'content_type': content_type,
                    'start_page': start_page,
                    'end_page': end_page
                })
                current_chunk = []
                content_type = 'text'
            is_code = True
            current_chunk.append(line)
            content_type = 'code'
        elif is_code and not line.startswith('>>>'):
            current_chunk.append(line)
        else:
            if is_code:
                chunks.append({
                    'id': str(uuid4()),
                    'pdf_name': pdf_name,
                    'version': version,
                    'doc_type': doc_type,
                    'section_number': section_num,
                    'section_path': section_num.split('.'),
                    'section_titles': [f"Section {'.'.join(section_num.split('.')[:i+1])}" 
                                    for i in range(len(section_num.split('.')) - 1)] + [section_title],
                    'content': '\n'.join(current_chunk),
                    'content_type': 'code',
                    'start_page': start_page,
                    'end_page': end_page
                })
                current_chunk = []
                is_code = False
                content_type = 'text'
            current_chunk.append(line)

    if current_chunk:
        chunks.append({
            'id': str(uuid4()),
            'pdf_name': pdf_name,
            'version': version,
            'doc_type': doc_type,
            'section_number': section_num,
            'section_path': section_num.split('.'),
            'section_titles': [f"Section {'.'.join(section_num.split('.')[:i+1])}" 
                            for i in range(len(section_num.split('.')) - 1)] + [section_title],
            'content': '\n'.join(current_chunk),
            'content_type': content_type,
            'start_page': start_page,
            'end_page': end_page
        })
    
    print(f"Section {section_num}: {len(chunks)} chunks created in {time() - start_time:.2f}s")
    return chunk_content(chunks)

def chunk_content(documents: List[Dict], max_tokens: int = 300) -> List[Dict]:
    """Split large text chunks."""
    chunked_docs = []
    for doc in documents:
        if doc['content_type'] == 'code':
            chunked_docs.append(doc)
        else:
            sentences = nltk.sent_tokenize(doc['content'])
            current_chunk = []
            current_tokens = 0
            for sentence in sentences:
                tokens = len(sentence.split())
                if current_tokens + tokens > max_tokens:
                    chunked_docs.append({
                        **doc,
                        'id': str(uuid4()),
                        'content': ' '.join(current_chunk),
                        'content_type': 'text'
                    })
                    current_chunk = [sentence]
                    current_tokens = tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += tokens
            if current_chunk:
                chunked_docs.append({
                    **doc,
                    'id': str(uuid4()),
                    'content': ' '.join(current_chunk),
                    'content_type': 'text'
                })
    return chunked_docs

def extract_section_content(hierarchy: Dict, cached_pages: Dict[int, str], pdf_name: str, version: str, doc_type: str) -> List[Dict]:
    """Extract content from cached pages for each section."""
    start_time = time()
    documents = []
    total_sections = sum(1 for _ in hierarchy.keys() for _ in [1] + list(hierarchy[_]['children'].keys()))
    print(f"Starting content extraction for {total_sections} sections")
    processed_sections = 0

    for section_num, section_data in hierarchy.items():
        if section_data['page_number'] is not None:
            start_page = section_data['page_number'] - 1
            next_page = None
            for other_num in sorted(hierarchy.keys()):
                if other_num > section_num and hierarchy[other_num]['page_number']:
                    next_page = hierarchy[other_num]['page_number'] - 1
                    break
            if not next_page:
                next_page = max(cached_pages.keys())

            content = []
            for page_num in range(start_page + 1, next_page + 1):
                if page_num in cached_pages and cached_pages[page_num]:
                    content.append(cached_pages[page_num])

            if content:
                title = section_data['title'] if section_data['title'] else f"Section {section_num}"
                chunks = segment_content(
                    '\n'.join(content),
                    section_num,
                    title,
                    start_page + 1,
                    next_page,
                    pdf_name,
                    version,
                    doc_type
                )
                documents.extend(chunks)

        if section_data['children']:
            child_docs = extract_section_content(hierarchy, cached_pages, pdf_name, version, doc_type)
            documents.extend(child_docs)

        processed_sections += 1
        if processed_sections % 10 == 0:
            print(f"Processed {processed_sections}/{total_sections} sections ({(processed_sections/total_sections)*100:.1f}%) in {time() - start_time:.2f}s")

    print(f"Content extraction completed in {time() - start_time:.2f}s, generated {len(documents)} documents")
    return documents

def main():
    """Process the PDF and generate JSON."""
    start_time = time()
    pdf_path = "../python_docs/python-3.13-docs/docs-pdf/library.pdf"
    pdf_name = pdf_path
    version = "3.13.2"
    doc_type = "Library Reference"

    with pdfplumber.open(pdf_path) as pdf:
        metadata = extract_pdf_metadata(pdf)
        cached_pages = cache_pages(pdf, metadata)
        index_pages = detect_index_pages(pdf)
        index_entries = parse_index(index_pages)
        hierarchy = build_hierarchy(index_entries)
        documents = extract_section_content(hierarchy, cached_pages, pdf_name, version, doc_type)

    with open('output.json', 'w') as f:
        json.dump({'documents': documents}, f, indent=2)
    print(f"Generated output.json with {len(documents)} documents in {time() - start_time:.2f}s")

if __name__ == '__main__':
    main()