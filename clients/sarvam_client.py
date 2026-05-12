import base64
import io
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class SarvamAPIError(RuntimeError):
    pass


@dataclass
class VoiceParams:
    speaker: str = "shubh"
    pace: float = 1.0
    model: str = "bulbul:v3"
    loudness: Optional[float] = None
    pitch: Optional[float] = None


class BaseSarvamClient:
    def translate(self, text: str, target_language: str) -> str:
        raise NotImplementedError

    def tts(self, text: str, language: str, voice_params: Dict) -> bytes:
        raise NotImplementedError

    def stt(
        self,
        audio_bytes: bytes,
        language: str,
        file_name: str = "question.wav",
        mime_type: str | None = None,
    ) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class RealSarvamClient(BaseSarvamClient):
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY", "").strip()
        self.base_url = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")
        self.chat_model = os.getenv("SARVAM_CHAT_MODEL", "sarvam-30b")
        self.translation_model = os.getenv("SARVAM_TRANSLATION_MODEL", "mayura:v1")
        self.source_language_code = os.getenv("SARVAM_SOURCE_LANGUAGE_CODE", "en-IN")
        self.translation_mode = os.getenv("SARVAM_TRANSLATION_MODE", "formal")
        self.translation_speaker_gender = os.getenv(
            "SARVAM_TRANSLATION_SPEAKER_GENDER", "Female"
        )
        self.tts_model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
        self.embeddings_url = os.getenv("SARVAM_EMBEDDINGS_URL", "").strip()
        self.timeout = int(os.getenv("SARVAM_TIMEOUT_SECONDS", "60"))
        self.translation_char_limit = (
            1000 if self.translation_model == "mayura:v1" else 2000
        )
        if not self.api_key:
            raise SarvamAPIError("SARVAM_API_KEY is missing. Add it to your .env file.")

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "api-subscription-key": self.api_key,
        }

    def _raise_for_status(self, response: requests.Response, feature: str):
        if response.ok:
            return
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        logger.error("Sarvam %s failed: %s", feature, payload)
        raise SarvamAPIError(f"Sarvam {feature} failed with status {response.status_code}.")

    @staticmethod
    def _safe_text(value) -> str:
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item or ""))
            return " ".join(part.strip() for part in parts if part).strip()
        return (value or "").strip()

    def translate(self, text: str, target_language: str) -> str:
        cleaned = self._safe_text(text)
        if not cleaned:
            return ""

        model = self.translation_model
        if target_language not in {
            "en-IN",
            "hi-IN",
            "bn-IN",
            "gu-IN",
            "kn-IN",
            "ml-IN",
            "mr-IN",
            "od-IN",
            "pa-IN",
            "ta-IN",
            "te-IN",
        }:
            model = "sarvam-translate:v1"

        payload = {
            "input": cleaned[: self.translation_char_limit],
            "source_language_code": self.source_language_code,
            "target_language_code": target_language,
            "speaker_gender": self.translation_speaker_gender,
            "mode": "formal" if model == "sarvam-translate:v1" else self.translation_mode,
            "model": model,
            "enable_preprocessing": True,
        }
        response = requests.post(
            f"{self.base_url}/translate",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.ok:
            return self._safe_text(response.json().get("translated_text"))

        fallback_payload = dict(payload)
        fallback_payload["source_language_code"] = "auto"
        fallback_response = requests.post(
            f"{self.base_url}/translate",
            headers={**self.headers, "Content-Type": "application/json"},
            json=fallback_payload,
            timeout=self.timeout,
        )
        self._raise_for_status(fallback_response, "translation")
        return self._safe_text(fallback_response.json().get("translated_text"))

    def tts(self, text: str, language: str, voice_params: Dict) -> bytes:
        cleaned_text = self._safe_text(text)
        if not cleaned_text:
            raise SarvamAPIError("Sarvam TTS received empty text.")
        payload = {
            "text": cleaned_text[:2500],
            "target_language_code": language,
            "speaker": voice_params.get("speaker", "shubh"),
            "pace": voice_params.get("pace", 1.0),
            "model": voice_params.get("model", self.tts_model),
            "output_audio_codec": "wav",
        }
        response = requests.post(
            f"{self.base_url}/text-to-speech",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "tts")
        audios = response.json().get("audios") or []
        if not audios:
            raise SarvamAPIError("Sarvam TTS returned no audio data.")
        audio_b64 = audios[0]
        return base64.b64decode(audio_b64)

    def stt(
        self,
        audio_bytes: bytes,
        language: str,
        file_name: str = "question.wav",
        mime_type: str | None = None,
    ) -> str:
        files = {
            "file": (
                file_name,
                audio_bytes,
                mime_type or "audio/wav",
            )
        }
        data = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language_code": language or "unknown",
        }
        response = requests.post(
            f"{self.base_url}/speech-to-text",
            headers=self.headers,
            files=files,
            data=data,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "stt")
        return response.json()["transcript"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.embeddings_url:
            response = requests.post(
                self.embeddings_url,
                headers={**self.headers, "Content-Type": "application/json"},
                json={"texts": texts},
                timeout=self.timeout,
            )
            self._raise_for_status(response, "embeddings")
            payload = response.json()
            if "vectors" in payload:
                return payload["vectors"]
        logger.warning(
            "Sarvam embeddings endpoint is not configured or unavailable; using TF-IDF fallback."
        )
        vectorizer = TfidfVectorizer(max_features=768)
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray().tolist()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response, "chat")
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise SarvamAPIError("Sarvam chat returned no choices.")
        message = choices[0].get("message") or {}
        content = self._safe_text(message.get("content"))
        if not content:
            raise SarvamAPIError("Sarvam chat returned empty content.")
        return content


class MockSarvamClient(BaseSarvamClient):
    def translate(self, text: str, target_language: str) -> str:
        return f"[Mock translation to {target_language}]\n{text}"

    def tts(self, text: str, language: str, voice_params: Dict) -> bytes:
        duration_seconds = min(max(len(text) // 25, 1), 8)
        sample_rate = 24000
        num_frames = duration_seconds * sample_rate
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            silent_frame = (0).to_bytes(2, byteorder="little", signed=True)
            wav_file.writeframes(silent_frame * num_frames)
        return buffer.getvalue()

    def stt(
        self,
        audio_bytes: bytes,
        language: str,
        file_name: str = "question.wav",
        mime_type: str | None = None,
    ) -> str:
        return f"Mock transcript generated for {file_name} in {language}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectorizer = TfidfVectorizer(max_features=256)
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray().tolist()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        trimmed = user_prompt[:700]
        return (
            "This is a mock Sarvam answer. Based on the retrieved chunks, the book discusses:\n\n"
            f"{trimmed}"
        )
