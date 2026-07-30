import json
import re
from pathlib import Path
from tqdm import tqdm

def count_tokens_approx(text: str) -> int:
    return len(text) // 4

def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def create_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens_approx(sentence)

        if sentence_tokens > chunk_size:
            if current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_token_count = 0
            chunks.append(sentence)
            continue

        if current_token_count + sentence_tokens > chunk_size:
            if current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))

            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = count_tokens_approx(s)
                if overlap_tokens + s_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences + [sentence]
            current_token_count = overlap_tokens + sentence_tokens
        else:
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))

    return chunks

def chunk_pages(pages: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    all_chunks = []
    chunk_id = 0

    print(f"\n✂️  Chunking {len(pages)} pages...")
    for page in tqdm(pages, desc="  Processing pages"):
        text = page.get('text', '').strip()
        if not text:
            continue

        page_chunks = create_chunks(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(page_chunks):
            if len(chunk_text.strip()) < 50:
                continue

            
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', page['manual_name'])
            chunk = {
            
                "chunk_id"            : f"{safe_name}_{chunk_id:06d}",
                "manual_name": page['manual_name'],
                "page_number": page['page_number'],
                "total_pages": page.get('total_pages', 0),
                "section_title": page.get('section_title', 'General'),
                "chunk_index": i,
                "total_chunks_in_page": len(page_chunks),
                "text": chunk_text,
                "token_count": count_tokens_approx(chunk_text),
                "char_count": len(chunk_text)
            }
            all_chunks.append(chunk)
            chunk_id += 1

    print(f"  ✅ Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks

def save_chunks(chunks: list, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(chunks)} chunks -> {output_path}")

def get_chunking_stats(chunks: list) -> dict:
    if not chunks:
        return {}
    token_counts = [c['token_count'] for c in chunks]
    manuals = list(set(c['manual_name'] for c in chunks))
    return {
        "total_chunks": len(chunks),
        "total_manuals": len(manuals),
        "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 0),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts)
    }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    processed_dir = BASE_DIR / "data/processed_chunks"
    json_files = list(processed_dir.glob("*_extracted.json"))

    if not json_files:
        print("❌ No extracted JSON files found. Run pdf_extractor.py first.")
    else:
        all_pages = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                pages = json.load(f)
                all_pages.extend(pages)
            print(f"📂 Loaded: {json_file.name} ({len(pages)} pages)")

        chunks = chunk_pages(all_pages, chunk_size=500, chunk_overlap=50)

        save_chunks(
            chunks,
            str(BASE_DIR / "data/processed_chunks/all_chunks.json")
        )

        stats = get_chunking_stats(chunks)
        print("\n📊 Chunking Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")