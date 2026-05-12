import hashlib


def pdf_cache_key(file_name: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{file_name}:{digest}"
