from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding = model.encode("Apa yang dimaksud mahasiswa aktif?")

print(type(embedding))
print(len(embedding))
