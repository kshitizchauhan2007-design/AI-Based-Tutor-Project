import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NCERT Class 8 Science AI Tutor",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #161B22;
}

.answer-box {
    background-color: #161B22;
    padding: 20px;
    border-radius: 12px;
    margin-top: 10px;
}

.source-box {
    background-color: #161B22;
    padding: 15px;
    border-left: 4px solid #58A6FF;
    border-radius: 8px;
    margin-top: 10px;
}

.info-box {
    background-color: #161B22;
    padding: 18px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"

# NOTE: we no longer load a .faiss file at all. Instead we rebuild the
# embedding matrix directly from the textbook documents at startup and
# search it with plain NumPy. This removes the "faiss" dependency
# completely, which is what was crashing the app.
DOCUMENTS_FILE = "class8_documents.joblib"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_resources():

    # Load embedding model
    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Load textbook documents
    documents = joblib.load(
        DOCUMENTS_FILE
    )

    # Build the text corpus in the same order as `documents`
    texts = [
        get_document_field(document, "text", "")
        for document in documents
    ]

    # Embed every chunk once, up front (cached for the app's lifetime)
    embedding_matrix = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embedding_matrix = np.asarray(
        embedding_matrix,
        dtype="float32"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    # Load FLAN-T5
    llm = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL
    )

    # Select CPU/GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    llm = llm.to(device)

    llm.eval()

    return (
        model,
        embedding_matrix,
        documents,
        tokenizer,
        llm,
        device
    )


# ============================================================
# GET DOCUMENT INFORMATION
# ============================================================

def get_document_field(
    document,
    field,
    default=""
):

    if isinstance(document, dict):

        return document.get(
            field,
            default
        )

    if hasattr(document, field):

        value = getattr(
            document,
            field
        )

        if value is not None:
            return value

    if hasattr(document, "meta"):

        metadata = document.meta or {}

        return metadata.get(
            field,
            default
        )

    return default


# ============================================================
# RETRIEVE TEXTBOOK DOCUMENTS (pure NumPy, no faiss)
# ============================================================

def retrieve_documents(
    question,
    model,
    embedding_matrix,
    documents,
    top_k=3
):

    # Create query embedding
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )[0]

    # Cosine similarity == dot product, since every vector is
    # already L2-normalized (normalize_embeddings=True above).
    scores = embedding_matrix @ query_embedding

    # Grab the top_k highest-scoring indices, best first
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    retrieved_documents = []

    for idx in top_indices:

        document = documents[idx]

        text = get_document_field(
            document,
            "text",
            ""
        )

        chapter = get_document_field(
            document,
            "chapter",
            "Unknown"
        )

        page = get_document_field(
            document,
            "printed_page",
            ""
        )

        if page == "":
            page = get_document_field(
                document,
                "page",
                "Unknown"
            )

        retrieved_documents.append({

            "text": str(text),

            "chapter": str(chapter),

            "page": str(page),

            "score": float(scores[idx])

        })

    return retrieved_documents


# ============================================================
# CREATE RAG PROMPT
# ============================================================

def create_prompt(
    question,
    retrieved_documents
):

    context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context += f"""

SOURCE {i}

Chapter:
{document["chapter"]}

Page:
{document["page"]}

Text:
{document["text"]}

"""


    prompt = f"""
You are an AI tutor for NCERT Class 8 Science.

Answer the student's question using ONLY the
textbook context provided below.

Rules:
- Use only information from the textbook context.
- Do not use outside knowledge.
- Do not invent information.
- Use simple Class 8 level language.
- Give a short and clear answer.
- If the answer is not present in the context,
  say that you could not find enough information
  in the textbook.

TEXTBOOK CONTEXT:

{context}

STUDENT QUESTION:

{question}

FINAL ANSWER:
"""

    return prompt


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    prompt,
    tokenizer,
    llm,
    device
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = llm.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# LOG INTERACTION
# ============================================================

def log_interaction(
    question,
    answer,
    retrieved_documents
):

    if retrieved_documents:

        best_similarity = (
            retrieved_documents[0]["score"]
        )

        chapters = "; ".join(
            dict.fromkeys(
                d["chapter"]
                for d in retrieved_documents
            )
        )

        pages = "; ".join(
            dict.fromkeys(
                d["page"]
                for d in retrieved_documents
            )
        )

    else:

        best_similarity = None
        chapters = ""
        pages = ""


    new_row = pd.DataFrame([{

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "question":
            question,

        "answer":
            answer,

        "best_similarity":
            best_similarity,

        "source_chapters":
            chapters,

        "source_pages":
            pages,

        "num_sources":
            len(retrieved_documents)

    }])


    file_name = "interaction_logs.csv"


    try:

        existing_data = pd.read_csv(
            file_name
        )

        updated_data = pd.concat(
            [existing_data, new_row],
            ignore_index=True
        )

    except FileNotFoundError:

        updated_data = new_row


    updated_data.to_csv(
        file_name,
        index=False
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    (
        model,
        embedding_matrix,
        documents,
        tokenizer,
        llm,
        device
    ) = load_resources()

except Exception as error:

    st.error(
        "❌ Unable to load the AI Tutor resources."
    )

    st.code(
        str(error)
    )

    st.info(
        "Make sure this file is in the same folder "
        "as app.py:\n\n"
        "class8_documents.joblib\n\n"
        "And that these packages are installed:\n"
        "streamlit, joblib, numpy, pandas, torch, "
        "sentence-transformers, transformers"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 AI Tutor")

    st.write(
        "NCERT Class 8 Science"
    )

    st.divider()

    st.write(
        f"📄 Documents: {len(documents)}"
    )

    st.write(
        f"🔎 Top-K Retrieval: {TOP_K}"
    )

    st.write(
        f"🎯 Threshold: {SIMILARITY_THRESHOLD}"
    )

    st.write(
        f"💻 Device: {device}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📚 NCERT Class 8 Science AI Tutor"
)

st.write(
    "Ask questions based strictly on the "
    "NCERT Class 8 Science textbook."
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="info-box">

        <b>Welcome to the NCERT Class 8 Science AI Tutor! 👋</b>

        <br><br>

        Try questions such as:

        <ul>
        <li>What is friction?</li>
        <li>What is photosynthesis?</li>
        <li>What is fermentation?</li>
        <li>What produces sound?</li>
        </ul>

        Answers are generated using retrieved
        NCERT textbook content.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]

            if sources:

                with st.expander(
                    "📖 Textbook Sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"""
                            <div class="source-box">

                            <b>Chapter:</b>
                            {source["chapter"]}

                            <br>

                            <b>Page:</b>
                            {source["page"]}

                            <br>

                            <b>Similarity:</b>
                            {source["score"]:.4f}

                            <br><br>

                            {source["snippet"]}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a Class 8 Science question..."
)


if question:

    # ========================================================
    # DISPLAY USER QUESTION
    # ========================================================

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # RETRIEVE
    # ========================================================

    with st.spinner(
        "🔎 Searching the NCERT textbook..."
    ):

        retrieved_documents = retrieve_documents(

            question,

            model,

            embedding_matrix,

            documents,

            TOP_K

        )


    # ========================================================
    # CHECK RELEVANCE
    # ========================================================

    if (

        not retrieved_documents

        or

        retrieved_documents[0]["score"]
        < SIMILARITY_THRESHOLD

    ):

        answer = (
            "I'm focused on Class 8 Science; "
            "I could not find enough information "
            "about this in the textbook. "
            "Try re-phrasing your question."
        )

    else:

        # ====================================================
        # CREATE PROMPT
        # ====================================================

        prompt = create_prompt(

            question,

            retrieved_documents

        )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.spinner(
            "🤖 Generating textbook-based answer..."
        ):

            answer = generate_answer(

                prompt,

                tokenizer,

                llm,

                device

            )


    # ========================================================
    # PREPARE SOURCES
    # ========================================================

    sources = []

    for document in retrieved_documents:

        sources.append({

            "chapter":
                document["chapter"],

            "page":
                document["page"],

            "score":
                document["score"],

            "snippet":
                document["text"][:500]

        })


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True
        )

        st.markdown(answer)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # SOURCES
        # ====================================================

        if sources:

            with st.expander(
                "📖 View textbook sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        <div class="source-box">

                        <b>Chapter:</b>
                        {source["chapter"]}

                        <br>

                        <b>Page:</b>
                        {source["page"]}

                        <br>

                        <b>Similarity:</b>
                        {source["score"]:.4f}

                        <br><br>

                        {source["snippet"]}...

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # ========================================================
    # SAVE TO SESSION
    # ========================================================

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })


    # ========================================================
    # LOG INTERACTION
    # ========================================================

    log_interaction(

        question,

        answer,

        retrieved_documents

    )
