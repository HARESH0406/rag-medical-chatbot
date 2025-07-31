import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import asyncio

# ✅ Ensure async event loop is available (Streamlit + gRPC fix)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load Google API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ✅ Streamlit UI
st.set_page_config(page_title="Dynamic Gemini PDF Chatbot", page_icon="🧠")
st.title("📄 Gemini Medical Chatbot (Dynamic PDF Upload)")

uploaded_file = st.file_uploader("Upload a medical PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # ✅ Load PDF and split more intelligently
    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # ✅ Create embeddings + FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)

    # ✅ Setup Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)

    # ✅ Create RAG chain with top 5 matching chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    # ✅ Chat interface
    query = st.text_input("Ask something about the uploaded document (e.g., drug dosage protocols):")
    if query:
        with st.spinner("🔍 Thinking..."):
            response = qa_chain(query)

        # ✅ Show answer
        st.write("### 🤖 Answer:")
        st.markdown(response["result"])

        # ✅ (Optional) Show which chunks were used — useful for debugging
        with st.expander("📄 Source Chunks (for Debugging)", expanded=False):
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(doc.page_content)
