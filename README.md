 # 🧠 Gemini-Powered Medical Chatbot

This project is a **Retrieval-Augmented Generation (RAG)**-based chatbot built using **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini 1.5 Flash (free tier)**. It allows users to upload medical PDFs and ask questions that are answered based on the content of the uploaded file — in real time.

---

## 🚀 Features

- 📄 Dynamic PDF Upload and Parsing  
- 🔗 LangChain + FAISS for Vector Search  
- 🔍 Google Gemini 1.5 Flash API for Answer Generation  
- 💡 Token-Efficient RAG Design  
- ✅ Runs Locally for Testing / Interviews  

---

## 🛠️ How to Run

1. **Create a `.env` file** in the project root and add your Gemini API key:
GOOGLE_API_KEY=your_api_key_here


2. **Install dependencies** (use virtual environment recommended):
bash
pip install -r requirements.txt

3. Start the Streamlit app:
streamlit run chatbot.py

👨‍💻 Author
Built by Haresh S.




