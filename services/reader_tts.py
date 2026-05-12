from typing import Dict, List

from clients.sarvam_client import BaseSarvamClient, SarvamAPIError
from utils.logger import get_logger

logger = get_logger(__name__)


class ReaderTTSService:
    def __init__(self, client: BaseSarvamClient, tts_char_limit: int = 2200):
        self.client = client
        self.tts_char_limit = tts_char_limit

    def collect_page_range(self, pages: List[Dict], start_page: int, end_page: int) -> str:
        selected = [
            page["text"]
            for page in pages
            if start_page <= page["page_number"] <= end_page
        ]
        return "\n\n".join(selected)

    def collect_first_n_pages(self, pages: List[Dict], n_pages: int) -> str:
        return "\n\n".join(page["text"] for page in pages[:n_pages])

    def prepare_text_for_narration(
        self,
        source_text: str | None,
        language_code: str,
        summary_prompt: str,
    ) -> str:
        text = (source_text or "").strip()
        if not text:
            raise ValueError("No narration text is available after translation.")
        if len(text) <= self.tts_char_limit:
            return text
        user_prompt = f"""
Target narration language: {language_code}
Please shorten this book excerpt for audio playback while preserving meaning.

Excerpt:
{text[:7000]}
""".strip()
        try:
            summarized = self.client.chat(summary_prompt, user_prompt)
        except SarvamAPIError as exc:
            logger.warning("Narration summarization failed, falling back to raw translated text: %s", exc)
            return text[: self.tts_char_limit]
        summarized_text = (summarized or "").strip()
        if not summarized_text:
            return text[: self.tts_char_limit]
        return summarized_text[: self.tts_char_limit]

    def generate_audio(self, text: str | None, language_code: str, voice_params: Dict) -> bytes:
        audio_text = (text or "").strip()
        if not audio_text:
            raise ValueError("No text is available for audio generation.")
        return self.client.tts(
            text=audio_text,
            language=language_code,
            voice_params=voice_params,
        )
