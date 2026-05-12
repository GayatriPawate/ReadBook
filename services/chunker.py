from typing import Dict, List


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: List[Dict], chunk_size: int = 900, overlap: int = 120) -> List[Dict]:
    chunks: List[Dict] = []
    chunk_id = 1
    for page in pages:
        page_chunks = _chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for text in page_chunks:
            chunks.append(
                {
                    "chunk_id": f"chunk-{chunk_id}",
                    "page_number": page["page_number"],
                    "page_range": f"{page['page_number']}",
                    "text": text,
                }
            )
            chunk_id += 1
    return chunks
