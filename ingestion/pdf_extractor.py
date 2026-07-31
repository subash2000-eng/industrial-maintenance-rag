"""
Phase 1 - PDF Extractor
Uses pypdf instead of PyMuPDF for cloud compatibility.
"""

import re
import json
from pathlib import Path
from tqdm import tqdm


def clean_text(text: str) -> str:
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_section_title(text: str) -> str:
    lines = text.strip().split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and len(line) < 100 and (
            line.isupper() or
            line.istitle() or
            re.match(r'^(\d+\.)+\s+\w+', line)
        ):
            return line
    return "General"


def extract_pdf(pdf_path: str) -> list:
    """
    Extract text from PDF using pypdf.
    Pure Python — works on all platforms.
    """
    from pypdf import PdfReader

    pdf_path  = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    manual_name = pdf_path.stem
    pages       = []

    print(f"\n📄 Extracting: {pdf_path.name}")

    reader      = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    for page_num in tqdm(
        range(total_pages), desc="  Pages"
    ):
        page     = reader.pages[page_num]
        raw_text = page.extract_text() or ""

        if len(raw_text.strip()) < 50:
            continue

        cleaned       = clean_text(raw_text)
        section_title = extract_section_title(cleaned)

        pages.append({
            "manual_name"  : manual_name,
            "page_number"  : page_num + 1,
            "total_pages"  : total_pages,
            "section_title": section_title,
            "text"         : cleaned,
            "char_count"   : len(cleaned)
        })

    print(
        f"  ✅ Extracted {len(pages)} pages "
        f"from {pdf_path.name}"
    )
    return pages


def extract_all_pdfs(
    manuals_dir: str,
    output_dir : str
) -> list:
    manuals_dir = Path(manuals_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(manuals_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in {manuals_dir}")
        return []

    print(f"\n🔍 Found {len(pdf_files)} PDF file(s)")
    all_pages = []

    for pdf_file in pdf_files:
        pages = extract_pdf(str(pdf_file))
        all_pages.extend(pages)

        output_file = output_dir / \
            f"{pdf_file.stem}_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved: {output_file.name}")

    print(f"\n✅ Total pages extracted: {len(all_pages)}")
    return all_pages


def get_extraction_stats(pages: list) -> dict:
    if not pages:
        return {}
    manuals    = list(set(p['manual_name'] for p in pages))
    total_chars = sum(p['char_count'] for p in pages)
    return {
        "total_manuals"    : len(manuals),
        "manual_names"     : manuals,
        "total_pages"      : len(pages),
        "total_characters" : total_chars,
        "avg_chars_per_page": round(
            total_chars / len(pages), 0
        )
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    pages    = extract_all_pdfs(
        manuals_dir=str(BASE_DIR / "data/raw_manuals"),
        output_dir =str(BASE_DIR / "data/processed_chunks")
    )
    if pages:
        stats = get_extraction_stats(pages)
        print("\n📊 Extraction Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")