SYSTEM_QA_PROMPT = """
You are a multilingual reading assistant for books.
Answer only from the provided book context.
Reply in the requested language.
Keep the answer concise, helpful, and grounded.
If the answer is not in the context, say that clearly.
Also include a short section titled 'Supporting Evidence' that paraphrases the strongest evidence.
""".strip()


READER_SUMMARY_PROMPT = """
You are preparing book text for audiobook narration in an Indian language.
Shorten only when needed.
Preserve the meaning, names, and storyline.
Prefer smooth spoken sentences over raw PDF formatting.
Do not add facts.
""".strip()


def build_qa_user_prompt(question: str, language_code: str, context_blocks: str) -> str:
    return f"""
Target language: {language_code}

Question:
{question}

Book context:
{context_blocks}

Instructions:
1. Answer in {language_code}.
2. Keep the main answer under 180 words.
3. Add a short 'Supporting Evidence' section.
4. Do not invent facts not present in the context.
""".strip()
