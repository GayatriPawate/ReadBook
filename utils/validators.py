import re
from pathlib import Path


SUPPORTED_LANGUAGES = {
    "Hindi": "hi-IN",
    "Kannada": "kn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Odia": "od-IN",
}

SUPPORTED_VOICES = {
    "Female - Shubh": {"speaker": "shubh", "pace": 1.0, "model": "bulbul:v3"},
    "Male - Aditya": {"speaker": "aditya", "pace": 1.0, "model": "bulbul:v3"},
    "Neutral - Rohan": {"speaker": "rohan", "pace": 1.0, "model": "bulbul:v3"},
}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}


def sanitize_text_input(text: str, max_length: int = 5000) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_length]


def validate_language_code(language_code: str) -> None:
    if language_code not in SUPPORTED_LANGUAGES.values():
        raise ValueError(f"Unsupported language: {language_code}")


def validate_pdf_upload(file_name: str, file_bytes: bytes) -> None:
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")
    if len(file_bytes) > 25 * 1024 * 1024:
        raise ValueError("PDF is too large for this demo. Please upload a file under 25 MB.")


def validate_audio_upload(file_name: str, file_bytes: bytes) -> None:
    if not file_bytes:
        raise ValueError("Uploaded audio is empty.")
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("Unsupported audio format. Use wav, mp3, m4a, or ogg.")


def validate_page_selection(start_page: int, end_page: int, total_pages: int) -> None:
    if start_page < 1 or end_page < 1:
        raise ValueError("Pages must start from 1.")
    if start_page > end_page:
        raise ValueError("Start page must be less than or equal to end page.")
    if end_page > total_pages:
        raise ValueError("Selected page range exceeds extracted pages.")
