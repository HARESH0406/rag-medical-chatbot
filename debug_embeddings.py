from langchain.embeddings import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=api_key
)

result = embeddings.embed_query("What is diabetes?")
print("✅ Embedding length:", len(result))







