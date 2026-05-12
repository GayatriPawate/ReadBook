from clients.sarvam_client import BaseSarvamClient


class VoiceSTTService:
    def __init__(self, client: BaseSarvamClient):
        self.client = client

    def transcribe(
        self,
        audio_bytes: bytes,
        file_name: str,
        language_code: str,
        mime_type: str | None = None,
    ) -> str:
        return self.client.stt(
            audio_bytes=audio_bytes,
            file_name=file_name,
            language=language_code,
            mime_type=mime_type,
        )
