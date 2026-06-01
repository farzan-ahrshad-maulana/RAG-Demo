import os

import chromadb
import fitz
from dotenv import load_dotenv
from google import genai


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()
    return text


text = extract_text("data/Buku_Peraturan_Akademik_2024_PR_25A.pdf")
# print(text[:1000])


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])

    return chunks


chunks = chunk_text(text)

# print("Jumlah chunk:", len(chunks))
# print()
# print(chunks[0])
#

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# print(type(gemini_client))
# print(dir(gemini_client))
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001", contents=text
    )
    return response.embeddings[0].values


embedding = get_embedding("Apa yang dimaksud siswa aktif?")
print(len(embedding))

chroma_client = chromadb.PersistentClient(path="./chroma.db")
try:
    chroma_client.delete_collection("itb_regulations")
except:
    pass
collection = chroma_client.get_or_create_collection(name="itb_regulations")

for idx, chunk in enumerate(chunks[:100]):
    #    print(f"Processing chunk {idx}")
    embedding = get_embedding(chunk)
    collection.add(ids=[str(idx)], documents=[chunk], embeddings=[embedding])
print("Jumlah dokumen:", collection.count())

question = """
Apa yang dimaksud mahasiswa aktif?
"""
question_embedding = get_embedding(question)
results = collection.query(query_embeddings=[question_embedding], n_results=3)
print(results["documents"][0])
context = "\n\n".join(results["documents"][0])

prompt = f"""
Jawablah pertanyaan berdasarkan context berikut.

Context:
{context}

Pertanyaan:
{question}
"""
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt
)
print(response.text)
