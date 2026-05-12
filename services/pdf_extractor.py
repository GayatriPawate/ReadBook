import io
from typing import Dict, List, Tuple

import pdfplumber

from utils.validators import sanitize_text_input


def extract_pdf_pages(file_name: str, file_bytes: bytes) -> Tuple[List[Dict], Dict]:
    pages: List[Dict] = []
    total_chars = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            cleaned = sanitize_text_input(text, max_length=50000)
            if cleaned.strip():
                pages.append(
                    {
                        "page_number": index,
                        "text": cleaned,
                        "char_count": len(cleaned),
                    }
                )
                total_chars += len(cleaned)

    if not pages:
        raise ValueError("No extractable text found in this PDF. Try a text-based PDF.")

    metadata = {
        "file_name": file_name,
        "total_pages_extracted": len(pages),
        "total_characters": total_chars,
        "average_chars_per_page": int(total_chars / max(len(pages), 1)),
    }
    return pages, metadata
