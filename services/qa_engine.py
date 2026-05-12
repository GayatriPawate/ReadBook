from typing import Callable, Dict, List

from clients.sarvam_client import BaseSarvamClient, SarvamAPIError


class QAEngine:
    def __init__(self, client: BaseSarvamClient):
        self.client = client

    def answer_question(
        self,
        question: str,
        language_code: str,
        retrieved_chunks: List[Dict],
        system_prompt: str,
        user_prompt_builder: Callable[[str, str, str], str],
    ) -> Dict:
        context_blocks = "\n\n".join(
            [
                f"[Chunk {chunk['chunk_id']} | Page {chunk['page_range']}]\n{chunk['text']}"
                for chunk in retrieved_chunks
            ]
        )
        prompt = user_prompt_builder(question, language_code, context_blocks)
        try:
            answer = self.client.chat(system_prompt, prompt)
        except SarvamAPIError:
            first_chunk = retrieved_chunks[0] if retrieved_chunks else {}
            fallback_excerpt = first_chunk.get("text", "")[:400]
            answer = (
                f"I could not generate a full answer from the chat API right now.\n\n"
                f"Closest supporting excerpt:\n{fallback_excerpt}"
            )
        return {
            "answer": answer,
            "sources": retrieved_chunks,
        }
