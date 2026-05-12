from typing import Callable, List

from clients.sarvam_client import BaseSarvamClient


class TranslatorService:
    def __init__(self, client: BaseSarvamClient, max_chars_per_call: int | None = None):
        self.client = client
        self.max_chars_per_call = max_chars_per_call or getattr(
            client,
            "translation_char_limit",
            950,
        )

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        if len(paragraph) <= self.max_chars_per_call:
            return [paragraph]

        words = paragraph.split()
        segments: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            addition = len(word) + 1
            if current_words and current_len + addition > self.max_chars_per_call:
                segments.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len += addition

        if current_words:
            segments.append(" ".join(current_words))

        return segments

    def split_text(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments: List[str] = []
        current = []
        current_len = 0
        for paragraph in paragraphs:
            paragraph_parts = self._split_large_paragraph(paragraph)
            for part in paragraph_parts:
                addition = len(part) + 2
                if current and current_len + addition > self.max_chars_per_call:
                    segments.append("\n\n".join(current))
                    current = [part]
                    current_len = len(part)
                else:
                    current.append(part)
                    current_len += addition
        if current:
            segments.append("\n\n".join(current))
        return segments

    def translate_document(
        self,
        text: str,
        target_language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        segments = self.split_text(text)
        if not segments:
            raise ValueError("No text is available for translation.")
        translated_segments = []
        total = len(segments)
        for index, segment in enumerate(segments, start=1):
            translated_text = self.client.translate(
                segment,
                target_language=target_language,
            )
            cleaned_translation = (translated_text or "").strip()
            if cleaned_translation:
                translated_segments.append(cleaned_translation)
            if progress_callback:
                progress_callback(index, total)
        if not translated_segments:
            raise ValueError("Translation returned empty text.")
        return "\n\n".join(translated_segments)
