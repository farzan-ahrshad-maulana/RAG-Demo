import chromadb
import fitz
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/Buku_Peraturan_Akademik_2024_PR_25A.pdf"


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    return embedding_model.encode(text).tolist()


print("Loading PDF...")
text = extract_text(PDF_PATH)

print("Chunking...")
chunks = chunk_text(text)

print(f"Jumlah chunk: {len(chunks)}")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

chroma_client = chromadb.PersistentClient(path="./chroma.db")

try:
    chroma_client.delete_collection("itb_regulations")
except:
    pass

collection = chroma_client.get_or_create_collection(name="itb_regulations")

print("Creating embeddings...")

for idx, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)

    collection.add(
        ids=[str(idx)],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[{"chunk_id": idx}],
    )

print("Indexing selesai")
print("Jumlah dokumen:", collection.count())
