from typing import Dict, List

import streamlit as st

from clients.sarvam_client import MockSarvamClient, RealSarvamClient, SarvamAPIError
from prompts.prompt_templates import (
    READER_SUMMARY_PROMPT,
    SYSTEM_QA_PROMPT,
    build_qa_user_prompt,
)
from services.chunker import chunk_pages
from services.pdf_extractor import extract_pdf_pages
from services.qa_engine import QAEngine
from services.reader_tts import ReaderTTSService
from services.retriever import Retriever
from services.translator import TranslatorService
from services.voice_stt import VoiceSTTService
from utils.logger import get_logger, log_feedback
from utils.validators import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VOICES,
    sanitize_text_input,
    validate_audio_upload,
    validate_language_code,
    validate_pdf_upload,
    validate_page_selection,
)


st.set_page_config(
    page_title="Read Any Book in Your Mother Tongue",
    page_icon="📚",
    layout="wide",
)

logger = get_logger(__name__)


def build_client(use_mock: bool):
    return MockSarvamClient() if use_mock else RealSarvamClient()


@st.cache_data(show_spinner=False)
def cached_extract_pdf(file_name: str, file_bytes: bytes):
    return extract_pdf_pages(file_name=file_name, file_bytes=file_bytes)


@st.cache_data(show_spinner=False)
def cached_chunk_pages(pages: List[Dict], chunk_size: int, overlap: int):
    return chunk_pages(pages=pages, chunk_size=chunk_size, overlap=overlap)


def render_feedback(feature_name: str, language_code: str):
    with st.expander(f"Feedback for {feature_name}", expanded=True):
        helpful = st.radio(
            f"Was the {feature_name.lower()} helpful?",
            ["Helpful 👍", "Not Helpful 👎"],
            horizontal=True,
            key=f"feedback_vote_{feature_name}",
        )
        comment = st.text_area(
            "Optional comment",
            placeholder="Tell us what worked well or what needs improvement.",
            key=f"feedback_comment_{feature_name}",
        )
        if st.button("Submit Feedback", key=f"feedback_submit_{feature_name}"):
            log_feedback(
                feature_used=feature_name,
                language=language_code,
                helpful=helpful,
                comment=sanitize_text_input(comment, max_length=1000),
            )
            st.success("Feedback saved locally.")


def init_session_state():
    defaults = {
        "pages": [],
        "chunks": [],
        "translated_text": "",
        "retriever": None,
        "qa_result": None,
        "narration_audio": None,
        "narration_text": "",
        "voice_transcript": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main():
    init_session_state()

    st.title("Read Any Book in Your Mother Tongue")
    st.caption(
        "Upload a book, translate it into an Indian language, listen to narration, and ask questions by voice."
    )

    with st.sidebar:
        st.header("Settings")
        target_language_label = st.selectbox(
            "Target language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0,
        )
        target_language_code = SUPPORTED_LANGUAGES[target_language_label]
        validate_language_code(target_language_code)

        qa_language_label = st.selectbox(
            "Q&A language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(target_language_label),
        )
        qa_language_code = SUPPORTED_LANGUAGES[qa_language_label]

        voice_option = st.selectbox("Voice", options=list(SUPPORTED_VOICES.keys()))
        reading_mode = st.radio(
            "Reading mode",
            options=["Excerpt", "Page Range", "First N pages"],
            index=2,
        )
        use_mock = st.toggle("Use Mock Sarvam APIs", value=False)
        chunk_size = st.slider("Chunk size", 500, 1500, 900, 100)
        overlap = st.slider("Chunk overlap", 50, 400, 120, 10)
        max_pages = st.slider("Max pages to process", 5, 100, 40, 5)

    try:
        client = build_client(use_mock)
    except SarvamAPIError as exc:
        st.warning(f"{exc} Falling back to mock mode for this session.")
        client = MockSarvamClient()
        use_mock = True
    translator = TranslatorService(client=client)
    reader = ReaderTTSService(client=client)
    stt_service = VoiceSTTService(client=client)
    qa_engine = QAEngine(client=client)

    uploaded_pdf = st.file_uploader("Upload a PDF book", type=["pdf"])

    if uploaded_pdf:
        file_bytes = uploaded_pdf.getvalue()
        try:
            validate_pdf_upload(uploaded_pdf.name, file_bytes)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.info(
            f"Loaded `{uploaded_pdf.name}` | Size: {len(file_bytes) / 1024:.1f} KB"
        )

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            extract_clicked = st.button("Extract Text", use_container_width=True)
        with col_b:
            translate_clicked = st.button("Translate", use_container_width=True)
        with col_c:
            audio_clicked = st.button("Generate Audio", use_container_width=True)
        with col_d:
            st.button("Ask by Voice", use_container_width=True, disabled=True)

        if extract_clicked:
            try:
                with st.spinner("Extracting text from PDF..."):
                    pages, metadata = cached_extract_pdf(
                        uploaded_pdf.name,
                        file_bytes,
                    )
                    pages = pages[:max_pages]
                    st.session_state.pages = pages
                    st.session_state.chunks = cached_chunk_pages(
                        pages=pages,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    st.session_state.retriever = Retriever.from_chunks(
                        chunks=st.session_state.chunks,
                        client=client,
                    )
                st.success(
                    f"Extracted {len(st.session_state.pages)} pages and built {len(st.session_state.chunks)} chunks."
                )
                st.json(metadata)
            except Exception as exc:
                logger.exception("PDF extraction failed")
                st.error(f"Extraction failed: {exc}")

        if translate_clicked:
            if not st.session_state.pages:
                st.warning("Extract text first.")
            else:
                try:
                    full_text = "\n\n".join(
                        f"[Page {page['page_number']}]\n{page['text']}"
                        for page in st.session_state.pages
                    )
                    progress = st.progress(0, text="Preparing translation...")

                    def on_progress(current: int, total: int):
                        progress.progress(
                            min(int((current / max(total, 1)) * 100), 100),
                            text=f"Translating segment {current}/{total}",
                        )

                    translated = translator.translate_document(
                        text=full_text,
                        target_language=target_language_code,
                        progress_callback=on_progress,
                    )
                    st.session_state.translated_text = translated
                    progress.progress(100, text="Translation complete")
                    st.success("Translation completed.")
                except Exception as exc:
                    logger.exception("Translation failed")
                    st.error(f"Translation failed: {exc}")

        narration_source_text = ""
        if st.session_state.pages:
            if reading_mode == "Excerpt":
                narration_source_text = st.text_area(
                    "Excerpt to narrate",
                    value=st.session_state.pages[0]["text"][:1500],
                    height=180,
                    help="Paste or edit a short excerpt. It will be translated, then narrated.",
                )
            elif reading_mode == "Page Range":
                range_col1, range_col2 = st.columns(2)
                with range_col1:
                    start_page = st.number_input("Start page", min_value=1, value=1)
                with range_col2:
                    end_page = st.number_input(
                        "End page",
                        min_value=1,
                        value=min(3, len(st.session_state.pages)),
                    )
                try:
                    validate_page_selection(
                        int(start_page), int(end_page), len(st.session_state.pages)
                    )
                    narration_source_text = reader.collect_page_range(
                        pages=st.session_state.pages,
                        start_page=int(start_page),
                        end_page=int(end_page),
                    )
                except Exception as exc:
                    st.error(str(exc))
            else:
                first_n = st.number_input(
                    "First N pages",
                    min_value=1,
                    max_value=max(1, len(st.session_state.pages)),
                    value=min(3, len(st.session_state.pages)),
                )
                narration_source_text = reader.collect_first_n_pages(
                    pages=st.session_state.pages,
                    n_pages=int(first_n),
                )

        if audio_clicked:
            if not narration_source_text.strip():
                st.warning("Provide a valid excerpt or page selection for narration.")
            else:
                try:
                    with st.spinner("Translating selected text for narration..."):
                        translated_for_audio = translator.translate_document(
                            text=narration_source_text,
                            target_language=target_language_code,
                        )
                    with st.spinner("Preparing narration text..."):
                        prepared_text = reader.prepare_text_for_narration(
                            source_text=translated_for_audio,
                            language_code=target_language_code,
                            summary_prompt=READER_SUMMARY_PROMPT,
                        )
                    with st.spinner("Generating audio with Sarvam TTS..."):
                        audio_bytes = reader.generate_audio(
                            text=prepared_text,
                            language_code=target_language_code,
                            voice_params=SUPPORTED_VOICES[voice_option],
                        )
                    st.session_state.narration_text = prepared_text
                    st.session_state.narration_audio = audio_bytes
                    st.success("Narration audio is ready.")
                except Exception as exc:
                    logger.exception("Narration failed")
                    st.error(f"Audio generation failed: {exc}")

        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Extracted Text Preview")
            if st.session_state.pages:
                query = st.text_input("Search extracted text")
                preview_pages = st.session_state.pages
                if query:
                    query = sanitize_text_input(query, max_length=120)
                    preview_pages = [
                        page
                        for page in preview_pages
                        if query.lower() in page["text"].lower()
                    ]
                for page in preview_pages[:5]:
                    st.markdown(f"**Page {page['page_number']}**")
                    st.text_area(
                        f"page_{page['page_number']}",
                        value=page["text"][:2500],
                        height=180,
                        label_visibility="collapsed",
                    )
            else:
                st.caption("No extracted text yet.")

        with right:
            st.subheader("Translated Text Preview")
            if st.session_state.translated_text:
                st.text_area(
                    "translated_preview",
                    value=st.session_state.translated_text[:5000],
                    height=420,
                    label_visibility="collapsed",
                )
            else:
                st.caption("No translated text yet.")

        if st.session_state.narration_audio:
            st.subheader("Narration")
            st.audio(st.session_state.narration_audio, format="audio/wav")
            with st.expander("Narration Text", expanded=False):
                st.write(st.session_state.narration_text)
            render_feedback("Narration", target_language_code)

        st.divider()
        st.subheader("Voice Q&A")
        st.caption("Record a question or upload an audio file, then get an answer with citations.")
        with st.form("qa_form"):
            recorded_audio = st.audio_input("Record your question")
            uploaded_audio = st.file_uploader(
                "Or upload an audio question",
                type=["wav", "mp3", "m4a", "ogg"],
                key="audio_question_uploader",
            )
            text_question = st.text_input("Fallback text question (optional)")
            answer_to_speech = st.checkbox("Read answer aloud", value=True)
            ask_voice_clicked = st.form_submit_button(
                "Submit Question",
                use_container_width=True,
            )

        if ask_voice_clicked:
            if not st.session_state.retriever or not st.session_state.chunks:
                st.warning("Extract text first so the app can build retrieval context.")
            else:
                audio_bytes = None
                audio_name = None
                audio_mime_type = None
                if recorded_audio is not None:
                    audio_bytes = recorded_audio.getvalue()
                    audio_name = "recorded_question.wav"
                    audio_mime_type = getattr(recorded_audio, "type", None) or "audio/wav"
                elif uploaded_audio is not None:
                    audio_bytes = uploaded_audio.getvalue()
                    audio_name = uploaded_audio.name
                    audio_mime_type = getattr(uploaded_audio, "type", None)

                transcript = ""
                if audio_bytes:
                    try:
                        validate_audio_upload(audio_name, audio_bytes)
                        with st.spinner("Transcribing voice question..."):
                            transcript = stt_service.transcribe(
                                audio_bytes=audio_bytes,
                                file_name=audio_name,
                                language_code=qa_language_code,
                                mime_type=audio_mime_type,
                            )
                    except Exception as exc:
                        st.error(f"Speech-to-text failed: {exc}")
                        transcript = ""
                elif text_question.strip():
                    transcript = sanitize_text_input(text_question, max_length=500)
                else:
                    st.warning("Provide a voice or text question.")
                    transcript = ""

                if transcript:
                    st.session_state.voice_transcript = transcript
                    try:
                        with st.spinner("Retrieving relevant book chunks..."):
                            retrieved = st.session_state.retriever.search(
                                query=transcript,
                                top_k=3,
                            )
                        with st.spinner("Generating answer..."):
                            result = qa_engine.answer_question(
                                question=transcript,
                                language_code=qa_language_code,
                                retrieved_chunks=retrieved,
                                system_prompt=SYSTEM_QA_PROMPT,
                                user_prompt_builder=build_qa_user_prompt,
                            )
                        answer_audio = None
                        if answer_to_speech:
                            with st.spinner("Generating spoken answer..."):
                                answer_audio = reader.generate_audio(
                                    text=result["answer"],
                                    language_code=qa_language_code,
                                    voice_params=SUPPORTED_VOICES[voice_option],
                                )
                        result["audio"] = answer_audio
                        st.session_state.qa_result = result
                        st.success("Answer generated successfully.")
                    except Exception as exc:
                        logger.exception("Q&A failed")
                        st.error(f"Q&A failed: {exc}")

        if st.session_state.qa_result:
            st.markdown("**Voice transcript**")
            st.write(st.session_state.voice_transcript)
            st.markdown("**Answer**")
            st.write(st.session_state.qa_result["answer"])
            if st.session_state.qa_result.get("audio"):
                st.audio(st.session_state.qa_result["audio"], format="audio/wav")
            st.markdown("**Supporting excerpts**")
            for item in st.session_state.qa_result["sources"]:
                st.info(
                    f"Chunk {item['chunk_id']} | Pages {item['page_range']} | Score {item['score']:.3f}\n\n{item['text'][:400]}"
                )
            render_feedback("Q&A", qa_language_code)

    else:
        st.caption("Upload a PDF to begin.")

    st.sidebar.divider()
    st.sidebar.caption("Secrets are loaded from `.env`. Feedback is logged to `data/feedback_log.jsonl`.")


if __name__ == "__main__":
    main()
