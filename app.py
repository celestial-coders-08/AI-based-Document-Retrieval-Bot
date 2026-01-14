import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# ENV (Local .env + Streamlit Secrets)
# --------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Document Q&A (LLaMA 3.2)",
    layout="wide"
)

st.markdown("""
<style>

/* -------- Reduce Heading Size -------- */
h1 {
    font-size: 1.7rem !important;
    font-weight: 600;
}

/* -------- Process Button -------- */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 0.35rem 0.8rem;
    font-size: 0.85rem;
    transition: all 0.25s ease-in-out;
}

.stButton > button:hover {
    background-color: #1e40af;
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 6px 16px rgba(37,99,235,0.45);
}

/* -------- Chat Message Hover -------- */
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    transition: transform 0.2s ease;
}

.stChatMessage:hover {
    transform: scale(1.01);
}

/* -------- Footer Text -------- */
.footer {
    text-align: center;
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def get_pdf_text(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(vectorstore):
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=400
    )

    llm = ChatHuggingFace(llm=endpoint)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say 'I don't know'."),
        ("human",
         "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.title("📄 AI-Based Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("❌ HF_TOKEN not found. Add it to .env or Streamlit Secrets.")
        st.stop()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("📤 Upload PDF")
        pdf = st.file_uploader("Upload PDF", type=["pdf"])

        if st.button("Process Document"):
            if not pdf:
                st.error("Please upload a PDF")
            else:
                with st.spinner("Processing document..."):
                    text = get_pdf_text(pdf)

                    if len(text.strip()) < 50:
                        st.error("PDF has no readable text (scanned PDF)")
                        st.stop()

                    chunks = split_text(text)
                    st.session_state.vectorstore = get_vectorstore(chunks)
                    st.success("✅ Document processed successfully")

    # ---------------- Chat History ----------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------- Chat Input ----------------
    if question := st.chat_input("Ask a question about the document"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            if not st.session_state.vectorstore:
                st.warning("Upload and process a document first.")
            else:
                with st.spinner("Thinking..."):
                    chain = get_chain(st.session_state.vectorstore)
                    response = chain.invoke(question)

                    answer = response.content if hasattr(response, "content") else str(response)
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

# --------------------------------------------------
if __name__ == "__main__":
    main()
